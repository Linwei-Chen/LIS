#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 7 13:57:50 2020 by Attila Lengyel - attila@lengyel.nl
Parameters:
- k : truncate filter after k*sigma, defines filter size (default: 3)
- init_sigma : initialization value for sigma (default: 1)
- use_cuda: whether or not to use gpu acceleration (default: True)
"""

# Import general dependencies
import math
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
import matplotlib.pyplot as plt
from PIL import Image
import os


class AdaDConv(nn.Module):
    """
    Adaptive-weighted downsampling
    """
    def __init__(self, in_channels, kernel_size=3, stride=2, groups=1, use_channel=True, use_nin=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size-1) // 2
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.use_channel = use_channel

        if use_nin:
            mid_channel = min((kernel_size ** 2 // 2), 4)
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * mid_channel , stride=stride,
                        kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * mid_channel), 
                nn.ReLU(True),
                nn.Conv2d(in_channels=groups * mid_channel, out_channels=groups * kernel_size ** 2, stride=1,
                        kernel_size=1, bias=False, padding=0, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            )

        else:
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride,
                        kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2), 
                # nn.Softmax(dim=1)
            )

        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Conv2d(in_channels=in_channels, out_channels= in_channels // 4, kernel_size=1, bias=False),
                # nn.BatchNorm2d(in_channels // 4), 
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4, out_channels = in_channels, kernel_size=1, bias=False),
                # nn.Sigmoid()
            )

        # nn.init.kaiming_normal_(self.channel_net[0].weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.weight_net[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, h, w = x.shape
        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1
        weight = self.weight_net(x) 
        _weight = weight
        weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow) 
        weight = weight.repeat(1, 1, c // self.groups, 1, 1, 1)

        if self.use_channel:
            tmp = self.channel_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
            # tmp[tmp < 1.] = tmp[tmp < 1.] ** 2
            # print(weight.shape)
            weight = weight * tmp
        weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
        weight = weight.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)

        pad_x = F.pad(x, pad=[self.pad] * 4, mode='reflect')
        # shape:  B x C x H // stride x W //stride x ksize x ksize
        pad_x = pad_x.unfold(2, self.kernel_size,self.stride).unfold(3, self.kernel_size, self.stride)
        pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
        res = weight * pad_x
        res = res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)
        return res
    
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, r=8):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // r, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // r, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        batch_size = x.size(0)  #   2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 150 x 150
        print(g_x.shape)
        g_x = g_x.permute(0, 2, 1)                                  #   2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)                                  #   2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       #   2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


def test_NonLocalBlockND():
    x = torch.rand(2, 4, 16, 16)
    m = nn.Sequential(
        NonLocalBlockND(in_channels=4, downsample_stride=2, sub_sample=True),
        nn.AvgPool2d(kernel_size=(1, 1), stride=2)
    )
    y = m(x)
    print(y.shape)
    pass

        
# ==================================
# ======== Gaussian filter =========
# ==================================

def gaussian_basis_filters(scale, gpu, k=3):
    std = torch.pow(2,scale)

    # Define the basis vector for the current scale
    filtersize = torch.ceil(k*std+0.5)
    x = torch.arange(start=-filtersize.item(), end=filtersize.item()+1)
    if gpu is not None: x = x.cuda(gpu); std = std.cuda(gpu)
    x = torch.meshgrid([x,x])

    # Calculate Gaussian filter base
    # Only exponent part of Gaussian function since it is normalized anyway
    g = torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    g = g / torch.sum(g)  # Normalize

    # Gaussian derivative dg/dx filter base
    dgdx = -x[0]/(std**3*2*math.pi)*torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    dgdx = dgdx / torch.sum(torch.abs(dgdx))  # Normalize

    # Gaussian derivative dg/dy filter base
    dgdy = -x[1]/(std**3*2*math.pi)*torch.exp(-(x[1]/std)**2/2)*torch.exp(-(x[0]/std)**2/2)
    dgdy = dgdy / torch.sum(torch.abs(dgdy))  # Normalize

    # Stack and expand dim
    basis_filter = torch.stack([g,dgdx,dgdy], dim=0)[:,None,:,:]

    return basis_filter


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, stride=2, padding=1):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        kernel = [
            [1/16., 1/8., 1/16.],
            [1/8., 1/4., 1/8.],
            [1/16., 1/8., 1/16.],
        ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # in_ch, out_ch,
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def __call__(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # self.device = x.device
        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels, stride=self.stride)
        return x

def test_GaussianBlurConv():
    x = torch.ones(1, 1, 3, 3)
    m = GaussianBlurConv(channels=1, stride=1, padding=1)
    # m = AdaPool(in_channels=64)
    # d = Downsample_PASA_group_softmax(in_channels=64, kernel_size=3, stride=2)
    y = m(x)
    # y = d(x)
    print(y.shape)
    print(y)


class LaplacianConv(nn.Module):
    def __init__(self, channels=3, stride=2, padding=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        kernel = [
            [1., 1., 1.],
            [1., -8., 1.],
            [1., 1., 1.],
        ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # in_ch, out_ch,
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def __call__(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # self.device = x.device
        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels, stride=self.stride)
        return x


def test_LaplacianConv():
    x = torch.ones(1, 1, 3, 3)
    m = LaplacianConv(channels=1, stride=1, padding=1)
    # m = AdaPool(in_channels=64)
    # d = Downsample_PASA_group_softmax(in_channels=64, kernel_size=3, stride=2)
    y = m(x)
    # y = d(x)
    print(y.shape)
    print(y)



class LLPFConv(nn.Module):
    """
    Learnable Low Pass Filter, Smooth-oriented convolution
    """
    def __init__(self, channels=3, stride=2, padding=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        kernel = [
            [1/16., 1/8., 1/16.],
            [1/8., 1/4., 1/8.],
            [1/16., 1/8., 1/16.],
        ]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # in_ch, out_ch,
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
 
    def __call__(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        # self.device = x.device
        x = F.conv2d(x, self.weight.reshape(self.channels, 1, -1).softmax(-1).reshape(self.channels, 1, 3, 3), 
                    padding=self.padding, groups=self.channels, stride=self.stride)
        return x


def test_LLPFConv():
    x = torch.ones(1, 1, 3, 3)
    m = LLPFConv(channels=1, stride=1, padding=1)
    # m = AdaPool(in_channels=64)
    # d = Downsample_PASA_group_softmax(in_channels=64, kernel_size=3, stride=2)
    y = m(x)
    # y = d(x)
    print(y.shape)
    print(y)


@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel


def BilateralFilter(batch_img, ksize, stride=1, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        
    if sigmaColor is None:
        sigmaColor = sigmaSpace
    
    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    
    # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, stride).unfold(3, ksize, stride)
    # print(patches.shape)
    patch_dim = patches.dim() # 6 
    # 求出像素亮度差
    diff_color = patches - batch_img[:, :, ::stride, ::stride].unsqueeze(-1).unsqueeze(-1)
    # 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)
    
    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)
    
    # 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
    # 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix


class BilateralFilterLayer(nn.Module):
    def __init__(self, ksize=3, stride=1, sigmaColor=None, sigmaSpace=None):
        super().__init__()
        if sigmaSpace is None:
            self.sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
            
        if sigmaColor is None:
            self.sigmaColor = self.sigmaSpace 
        
        self.pad = (ksize - 1) // 2
        self.ksize = ksize
        self.stride = stride


    def forward(self, batch_img):
        # return BilateralFilter(batch_img, self.ksize, self.stride, sigmaColor=None, sigmaSpace=None)
        device = batch_img.device
        batch_img_pad = F.pad(batch_img, pad=[self.pad] * 4, mode='reflect')
        
        # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
        # patches.shape:  B x C x H x W x ksize x ksize
        patches = batch_img_pad.unfold(2, self.ksize, self.stride).unfold(3, self.ksize, self.stride)
        # print(patches.shape)
        patch_dim = patches.dim() # 6 
        # 求出像素亮度差
        diff_color = patches - batch_img[:, :, ::self.stride, ::self.stride].unsqueeze(-1).unsqueeze(-1)
        # 根据像素亮度差，计算权重矩阵
        weights_color = torch.exp(-(diff_color ** 2) / (2 * self.sigmaColor ** 2))
        # 归一化权重矩阵
        weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)
        
        # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
        weights_space = getGaussianKernel(self.ksize, self.sigmaSpace).to(device)
        weights_space_dim = (patch_dim - 2) * (1,) + (self.ksize, self.ksize)
        weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)
        
        # 两个权重矩阵相乘得到总的权重矩阵
        weights = weights_space * weights_color
        # 总权重矩阵的归一化参数
        weights_sum = weights.sum(dim=(-1, -2))
        # 加权平均
        weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
        return weighted_pix

class BilateralCosFilterLayer(BilateralFilterLayer):
    def __init__(self, ksize=3, stride=1, sigmaColor=None, sigmaSpace=None):
        super().__init__(ksize=ksize, stride=stride, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)


    def forward(self, batch_img):
        device = batch_img.device

        batch_img_pad = F.pad(batch_img, pad=[self.pad] * 4, mode='reflect')
        
        # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
        # patches.shape:  B x C x H x W x ksize x ksize
        patches = batch_img_pad.unfold(2, self.ksize , self.stride).unfold(3, self.ksize, self.stride)
        # print(patches.shape)
        patch_dim = patches.dim() # 6 
        # 求出像素亮度差
        # diff_color = patches - batch_img[:, :, ::self.stride, ::self.stride].unsqueeze(-1).unsqueeze(-1)
        # diff_color = patches * batch_img[:, :, ::self.stride, ::self.stride].unsqueeze(-1).unsqueeze(-1)
        diff_color = torch.nn.CosineSimilarity(patches, batch_img[:, :, ::self.stride, ::self.stride].unsqueeze(-1).unsqueeze(-1)).unsqueeze(1)
        # 根据像素亮度差，计算权重矩阵
        weights_color = torch.exp(-(diff_color ** 2) / (2 * self.sigmaColor ** 2))
        # 归一化权重矩阵
        weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)
        
        # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
        weights_space = getGaussianKernel(self.ksize, self.sigmaSpace).to(device)
        weights_space_dim = (patch_dim - 2) * (1,) + (self.ksize, self.ksize)
        weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)
        
        # 两个权重矩阵相乘得到总的权重矩阵
        weights = weights_space * weights_color
        # 总权重矩阵的归一化参数
        weights_sum = weights.sum(dim=(-1, -2))
        # 加权平均
        weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
        return weighted_pix

def feature2image(feat, save_dir, name, normalize=False):
    # colors = plt.cm.jet(np.linspace(0.0, 1.00, 256))
    colors = plt.cm.viridis(np.linspace(0.0, 1.00, 256))

    _, _, h, w = feat.shape
    feat = feat.mean(dim=1).repeat(3, 1, 1)
    # feat = F.interpolate(feat[None, ], scale_factor=1, mode='bilinear', align_corners=True)[0]
    feat = F.interpolate(feat[None, ], size=(400, 600), mode='nearest')[0]
    # feat = F.interpolate(feat[None, ], size=(400, 600), mode='bilinear')[0]
    # print(vis_feature.shape)
    # std = torch.tensor([103.530, 116.280, 123.675])[None, :, None, None].to(img.device)
    # print(img_metas)
    # name = np.random.randint(low=0, high=100000)
    # name = os.path.splitext(os.path.split(img_metas[0]['ori_filename'])[1])[0]
    save_img = feat.permute(1, 2, 0).detach().cpu().numpy()
    if normalize:
        save_img = (save_img - save_img.min()) / (save_img.max() - save_img.min()) * 255
    else:
        save_img = save_img * 255
        save_img = np.clip(save_img, 0, 255)
    save_img = save_img.astype(np.uint8)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    colored = np.zeros(shape=(save_img.shape[0], save_img.shape[1], 3)).astype(np.uint8)
    for i in range(3):
        for num in np.unique(save_img):
            # print(num)
            # print(colored[save_img == int(num)])
            colored[:,:,i][save_img[:, :, i] == int(num)] = colors[int(num)][i] * 255.
    # Image.fromarray(save_img).save(f'{save_dir}/{name}_.jpg')
    Image.fromarray(colored).save(f'{save_dir}/{name}.png')
    pass

def tensor2img(t, save_dir, name):
    norm = torch.tensor([103.530, 116.280, 123.675])[None, :, None, None].to(t.device)
    t += norm
    img = t[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(img).save(f'{save_dir}/{name}.png')

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

if __name__ == '__main__':
    # x = torch.rand(2, 64, 16, 16)
    # y = nn.MaxPool2d(kernel_size=2, stride=1)(x)
    # y = BilateralFilterLayer()(x)
    # y = PPFuse(in_dim=64)(x)
    # print(y.shape)
    # print(y)
    # y = GaussianBlurConv()(x)
    # print(y.shape)
    # test_AdaDConv()
    # test_GaussianBlurConv()
    # test_LaplacianConv()
    # test_LLPFConv()
    test_NonLocalBlockND()