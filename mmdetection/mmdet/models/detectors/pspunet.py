import sys
import os.path as osp
sys.path.append(osp.join(osp.abspath(__file__), '../../'))
sys.path.append('/home/ubuntu/code/LowLight/LPTN/codes/models')

import math
from codes.models.archs.noisemodel import NoiseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models



class ConvBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 k=1,
                 s=1,
                 p=0,
                 d=1,
                 pad_type='reflect',
                 norm='none',
                 sn=True,
                 activation='leaky_relu',
                 deconv=False):
        super(ConvBlock, self).__init__()

        layers = []
        # Conv
        if deconv is True:
            if sn is True:
                layers += [
                    nn.utils.spectral_norm(
                        nn.ConvTranspose2d(in_dim,
                                           out_dim,
                                           k,
                                           s,
                                           padding=p,
                                           padding_mode='zeros',
                                           dilation=d,
                                           bias=False))
                ]
            else:
                layers += [
                    nn.ConvTranspose2d(in_dim,
                                       out_dim,
                                       k,
                                       s,
                                       padding=p,
                                       padding_mode='zeros',
                                       dilation=d,
                                       bias=False)
                ]
        else:
            if sn is True:
                layers += [
                    nn.utils.spectral_norm(
                        nn.Conv2d(in_dim,
                                  out_dim,
                                  k,
                                  s,
                                  padding=p,
                                  padding_mode=pad_type,
                                  dilation=d,
                                  bias=False))
                ]
            else:
                layers += [
                    nn.Conv2d(in_dim,
                              out_dim,
                              k,
                              s,
                              padding=p,
                              padding_mode=pad_type,
                              dilation=d,
                              bias=False)
                ]
        # Norm
        if norm == 'bn':
            layers += [nn.BatchNorm2d(out_dim, affine=True)]
        elif norm == 'inn':
            layers += [nn.InstanceNorm2d(out_dim, affine=True)]

        # Activation
        if activation == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif activation == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif activation == 'sigmoid':
            layers += [nn.Sigmoid()]
        elif activation == 'tanh':
            layers += [nn.Tanh()]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class DeConvBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 k=1,
                 s=1,
                 p=0,
                 d=1,
                 pad_type='reflect',
                 norm='none',
                 sn=True,
                 activation='leaky_relu'):
        super(DeConvBlock, self).__init__()

        self.conv2d = ConvBlock(in_dim,
                                out_dim,
                                k=k,
                                s=s,
                                p=p,
                                d=d,
                                pad_type=pad_type,
                                norm=norm,
                                sn=sn,
                                activation=activation,
                                deconv=True)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResnetBlock(nn.Module):
    def __init__(self,
                 dim,
                 k=3,
                 s=1,
                 p=1,
                 d=1,
                 pad_type='reflect',
                 norm='none',
                 sn=True,
                 activation='none',
                 use_dropout=False):
        super(ResnetBlock, self).__init__()

        layers = [
            ConvBlock(dim,
                      dim,
                      k=k,
                      s=s,
                      p=p,
                      d=d,
                      pad_type=pad_type,
                      norm=norm,
                      sn=sn,
                      activation=activation)
        ]
        if use_dropout:
            layers += [nn.Dropout(p=0.5)]
        layers += [
            ConvBlock(dim,
                      dim,
                      k=k,
                      s=s,
                      p=p,
                      d=d,
                      pad_type=pad_type,
                      norm='none',
                      sn=sn,
                      activation='none')
        ]
        self.res_block = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.res_block(x)
        out = out + residual
        return out


class PyramidPooling(nn.Module):
    def __init__(self,
                 in_channels,
                 pool_sizes,
                 norm,
                 activation,
                 pad_type='reflect'):
        super(PyramidPooling, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(
                ConvBlock(in_channels,
                          int(in_channels / len(pool_sizes)),
                          k=1,
                          s=1,
                          p=0,
                          pad_type=pad_type,
                          norm=norm,
                          activation=activation))
        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]

        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            # out = F.avg_pool2d(x, int(h / pool_size), int(h / pool_size), 0)
            out = F.adaptive_avg_pool2d(x, (h, h))
            out = module(out)
            out = F.interpolate(out, size=(h, w), mode='bilinear')
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)


class NoiseGenerator(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.use_noise_model = kargs['use_noise_model']
        self.decompose_noise = kargs['decompose_noise']

        ch = 64
        norm_g = 'inn'
        residual_block = 9

        self.SE = nn.Sequential(
            ConvBlock(4, ch, k=7, s=1, p=3),
            ConvBlock(ch, ch * 2, k=4, s=2, p=1, norm=norm_g),
            ConvBlock(ch * 2, ch * 4, k=4, s=2, p=1, norm=norm_g),
            ConvBlock(ch * 4, ch * 8, k=4, s=2, p=1, norm=norm_g),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.C = nn.Linear(ch * 8, 5)


        self.c1 = ConvBlock(3, 64, k=4, s=2, p=1)
        self.c2 = ConvBlock(64, 128, k=4, s=2, p=1, norm=norm_g)
        self.c3 = ConvBlock(128, 256, k=4, s=2, p=1, norm=norm_g)
        self.c4 = ConvBlock(256, 512, k=4, s=2, p=1, norm=norm_g)
        self.c5 = ConvBlock(512, 512, k=4, s=2, p=1, norm=norm_g)

        # residual_list = []
        # for i in range(residual_block):
        #   residual_list += [ResnetBlock(512, k=3, s=1, p=1, norm=norm_g, use_dropout=False)]
        #self.RES = nn.Sequential(*residual_list)

        self.dc1 = DeConvBlock(512, 512, k=4, s=2, p=1, norm=norm_g)
        # self.dc1 = DeConvBlock(512, 512, k=4, s=2, p=1, norm=norm_g)
        self.dc2 = DeConvBlock(1024, 256, k=4, s=2, p=1, norm=norm_g)
        self.dc3 = DeConvBlock(512, 128, k=4, s=2, p=1, norm=norm_g)
        self.dc4 = DeConvBlock(256, 64, k=4, s=2, p=1, norm=norm_g)
        self.ppool = PyramidPooling(128, [6, 3, 2, 1], norm=norm_g,activation='leaky_relu')
        self.dc5 = DeConvBlock(128 * 2, 128, k=4, s=2, p=1, norm=norm_g,)

        self.final = ConvBlock(128, 3, k=1, s=1, p=0, norm=norm_g, activation=None)

    def forward(self, c):
        _, _, ih, iw = c.shape
        rh, rw = int(math.ceil(ih / 32) * 32), int(math.ceil(iw / 32) * 32)
        rc = F.interpolate(c,
                           size=(rh, rw),
                           mode='bilinear',
                           align_corners=True)

        oc1 = self.c1(rc)
        oc2 = self.c2(oc1)
        oc3 = self.c3(oc2)
        oc4 = self.c4(oc3)
        oc5 = self.c5(oc4)
        odc1 = self.dc1(oc5, dim=1)
        # odc1 = self.dc1(oc5)
        odc2 = self.dc2(torch.cat([odc1, oc4], dim=1))
        odc3 = self.dc3(torch.cat([odc2, oc3], dim=1))
        odc4 = self.dc4(torch.cat([odc3, oc2], dim=1))
        # print(odc4.shape, oc1.shape)
        odc5 = self.ppool(torch.cat([odc4, oc1], dim=1))
        out = self.dc5(odc5)
        # print(out.shape, z.shape, c.shape)
        # out = out + z + c
        # out = out * 0.2 + c

        # return out, latent_a, latent_p, latent_n
        out = self.final(out)
        out = F.interpolate(out, size=(ih, iw),
                            mode='bilinear', align_corners=True)
        return out


if __name__ == '__main__':
    M = NoiseGenerator(use_noise_model=True, decompose_noise=True).cuda()
    x = torch.rand(2, 3, 1200, 800).cuda()
    y = M(x).cuda()
    print(y, y.shape)
