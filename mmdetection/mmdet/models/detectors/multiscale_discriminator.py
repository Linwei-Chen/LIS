import re
from mmcv.runner import BaseModule, auto_fp16
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.modules.pooling import AdaptiveAvgPool2d

class DiscriminatorFeaturesLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.l1 = nn.DataParallel(nn.L1Loss())
        self.l1 = nn.L1Loss()

    def forward(self, ds_fake, ds_real):
        """
        :param ds_fake: [D1:[layer1_outs, layer2_outs ...], D2, D3]
        :param ds_real:
        :return:
        """
        loss = 0

        for scale in range(len(ds_real)):
            # last is D_outs, do not use as features
            for l in range(len(ds_real[scale]) - 1):
                loss += self.l1(ds_fake[scale][l], ds_real[scale][l].detach())
        return loss / float(len(ds_real))


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            # self.loss = nn.DataParallel(nn.MSELoss())
            self.loss = nn.MSELoss()
        else:
            # self.loss = nn.DataParallel(nn.BCELoss())
            self.loss = nn.BCELoss()
        print(f'===> {self.__class__.__name__} | use_lsgan:{use_lsgan} | loss:{self.loss}')

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.shape).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.shape).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real).to(pred.device)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real).to(input[-1].device)
            return self.loss(input[-1], target_tensor)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=True, use_psp=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)
        self.use_psp = use_psp
        if self.use_psp:
            self.downsample = [
                # nn.AdaptiveAvgPool2d((11, 11)),
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.AdaptiveAvgPool2d((3, 3)),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.AdaptiveAvgPool2d((1, 1)), # 
            ]
        else:
            self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.dfloss = DiscriminatorFeaturesLoss()
        self.ganloss = GANLoss()

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                if self.use_psp:
                    input_downsampled = self.downsample[i](input)
                else:
                    input_downsampled = self.downsample(input_downsampled)
        return result

    def D_loss(self, fake, real):
        fake_res = self.forward(fake.detach())
        real_res = self.forward(real.detach())

        fakeloss = self.ganloss(fake_res, target_is_real=False)
        realloss = self.ganloss(real_res, target_is_real=True)
        d_loss = 0.5 * (fakeloss + realloss)

        fake_res = self.forward(fake)
        ganloss = self.ganloss(fake_res, target_is_real=True)
        dfloss = self.dfloss(fake_res, real_res)
        # print(fake_res, len(fake_res))
        # print(real_res, len(real_res), len(real_res[0]))
        # print(d_loss)
        # print(ganloss)
        # print(dfloss)

        losses = dict()
        losses['d_loss'] = d_loss
        losses['ganloss'] = ganloss
        losses['dfloss'] = dfloss

        return losses

    def FPN_D_loss(self, fakes, reals):
        losses = dict()
        losses['d_loss'] = 0
        losses['ganloss'] = 0
        losses['dfloss'] = 0
        n = len(fakes)
        for fake, real in zip(fakes, reals):
            temp_losses = self.D_loss(fake, real)
            losses['d_loss'] += temp_losses['d_loss']
            losses['ganloss'] += temp_losses['ganloss']
            losses['dfloss'] += temp_losses['dfloss']
        
        losses['d_loss'] /= n
        losses['ganloss'] /= n
        losses['dfloss'] /= n
        return losses

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

def test_discriminator():
    pass
    D = MultiscaleDiscriminator(input_nc=256, getIntermFeat=True)
    x = torch.rand(2, 256, 128, 128)
    y = D(x)
    for i in y:
        print(i[0].shape)

if __name__ == '__main__':
    test_discriminator()