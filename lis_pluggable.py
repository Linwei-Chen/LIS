# -*- coding: utf-8 -*-
"""
lis_pluggable.py — LIS《Instance Segmentation in the Dark》(IJCV 2023) 即插即用模块
================================================================================

import 本文件即可把论文全部自定义组件注册进 mmdet 2.15.1，无需替换任何 mmdet 源码：

    pip install mmdet==2.15.1 mmcv-full==1.4.7   # torch 1.8~1.12 + cudatoolkit 对应版本
    python -c "import lis_pluggable"             # 冒烟测试

之后正常使用 Configs/ 下的训练/测试配置（详见 README）。

注册的组件（与 IJCV 2023 论文发布权重逐位对齐）：
  BACKBONES  ResNetAdaD / ResNetAdaDSmoothPrior     — AWD（自适应加权下采样）+ SCB（smooth-oriented 卷积块）
             ConvNeXtAdaD / ConvNeXtAdaDSmoothPrior — 同上，ConvNeXt 变体（mmdet 2.15.1 官方无 ConvNeXt，随本文件内置）
  NECKS      FPN（覆盖注册）                        — forward 额外支持 return_backbone（DSL 训练需要）
  DETECTORS  MaskRCNNNoiseInv                       — DSL（disturbance suppression learning）：clean/noisy 双前向 + 特征一致性
  TRANSFORMS AddNoisyImg / NoiseModel / HistMatch   — RAW 低光噪声合成管线（PGRU 相机噪声模型）

说明：
  * 相机噪声参数位于本文件同目录的 camera_params/（CanonEOS5D4）。
  * PointRend / Mask2Former 变体与 faster RCNN 版本请使用仓库内 mmdetection/（完整改动版 2.15.1）复现。
  * 本文件由仓库工具脚本从 mmdetection/ fork 自动抽取生成，勿手改。
"""

import copy
from abc import ABCMeta, abstractmethod
import cv2
import math
import os
import os.path as osp
import random
import warnings
from functools import partial
from itertools import chain
from os.path import join
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.checkpoint import checkpoint as cp

from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks import (NORM_LAYERS, DropPath, build_activation_layer,
                             build_norm_layer)
from mmcv.runner import BaseModule, auto_fp16, load_checkpoint
from mmcv.runner.base_module import ModuleList, Sequential

from mmdet.datasets.builder import PIPELINES
from mmdet.models.backbones.resnet import Bottleneck, ResNet
from mmdet.models.builder import BACKBONES, DETECTORS, NECKS
from mmdet.models.detectors.mask_rcnn import MaskRCNN
from mmdet.models.necks.fpn import FPN as _OfficialFPN

colors = plt.cm.jet(np.linspace(0.0, 1.00, 256))


# ============================== 自定义卷积（AWD / SCB 先验滤波）: from backbones/CustomConv.py
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


# ============================== ConvNeXt backbone 家族（官方 2.15.1 无 ConvNeXt，整体内置）: from backbones/convnext.py
class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def __init__(self, init_cfg=None):
        super(BaseBackbone, self).__init__(init_cfg)

    @abstractmethod
    def forward(self, x):
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(BaseBackbone, self).train(mode)

@NORM_LAYERS.register_module('LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)

class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 pretrained=None
                 ):
        super().__init__()
        self.pretrained=pretrained
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)
        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)
        x = self.norm(x)

        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)

        if self.linear_pw_conv:
            x = x.permute(0, 3, 1, 2)  # permute back

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = shortcut + self.drop_path(x)
        return x

class SmoothPriorConvNeXtBlock(ConvNeXtBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        in_channels = kwargs['in_channels']
        self.llpf = LLPFConv(channels=in_channels, stride=1, padding=1)
        self.learnable_conv1 = build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    bias=False)

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x) + self.learnable_conv1(self.llpf(x))
        x = self.norm(x)

        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)

        if self.linear_pw_conv:
            x = x.permute(0, 3, 1, 2)  # permute back

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = shortcut + self.drop_path(x)
        return x

@BACKBONES.register_module()
class ConvNeXt(BaseBackbone):
    """ConvNeXt.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0])[1],
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            # print('stage:', i)
            # print('pre:', x.shape)
            x = self.downsample_layers[i](x)
            # print('post:', x.shape)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()

@BACKBONES.register_module()
class ConvNeXtAdaD(ConvNeXt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sampler_channels = self.arch_settings[kwargs['arch']]['channels'][:3]
        self.AdaD = nn.ModuleList([
            nn.Identity(),
            AdaDConv(in_channels=sampler_channels[0], kernel_size=3, stride=1, groups=sampler_channels[0]),
            AdaDConv(in_channels=sampler_channels[1], kernel_size=3, stride=1, groups=sampler_channels[1]),
            AdaDConv(in_channels=sampler_channels[2], kernel_size=3, stride=1, groups=sampler_channels[2]),
        ])

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            if i == 0:
                pass
            elif i in (1, 2, 3):
                # _, _, h, w = x.shape
                x = self.AdaD[i](x)
            else:
                pass

            # print('stage:', i)
            # print('pre:', x.shape)
            x = self.downsample_layers[i](x)
            # print('post:', x.shape)
            x = stage(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        return tuple(outs)

@BACKBONES.register_module()
class ConvNeXtAdaDSmoothPrior(ConvNeXt):
    def __init__(self,
                arch='tiny',
                in_channels=3,
                stem_patch_size=4,
                norm_cfg=dict(type='LN2d', eps=1e-6),
                act_cfg=dict(type='GELU'),
                linear_pw_conv=True,
                drop_path_rate=0.,
                layer_scale_init_value=1e-6,
                out_indices=-1,
                frozen_stages=0,
                gap_before_final_norm=True,
                init_cfg=None):
        super().__init__(arch,
                        in_channels,
                        stem_patch_size,
                        norm_cfg,
                        act_cfg,
                        linear_pw_conv,
                        drop_path_rate,
                        layer_scale_init_value,
                        out_indices,
                        frozen_stages,
                        gap_before_final_norm,
                        init_cfg)
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0
        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0])[1],
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                SmoothPriorConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()
        sampler_channels = self.arch_settings[arch]['channels'][:3]
        self.AdaD = nn.ModuleList([
            nn.Identity(),
            AdaDConv(in_channels=sampler_channels[0], kernel_size=3, stride=1, groups=sampler_channels[0]),
            AdaDConv(in_channels=sampler_channels[1], kernel_size=3, stride=1, groups=sampler_channels[1]),
            AdaDConv(in_channels=sampler_channels[2], kernel_size=3, stride=1, groups=sampler_channels[2]),
        ])

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            if i == 0:
                pass
            elif i in (1, 2, 3):
                # _, _, h, w = x.shape
                x = self.AdaD[i](x)
            else:
                pass

            # print('stage:', i)
            # print('pre:', x.shape)
            x = self.downsample_layers[i](x)
            # print('post:', x.shape)
            x = stage(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        return tuple(outs)


# ============================== SCB Bottleneck（及其依赖 BasicBlock）: from backbones/resnet.py
class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class SmoothPriorBottleneck(Bottleneck):
    def __init__(self, 
                # filter_priors=['gaussian'], 
                # filter_priors=['gaussian', 'mean'], 
                # filter_priors=['gaussian', 'mean', 'laplacian'], 
                # filter_priors=['gaussian', 'mean', 'learnable'], 
                filter_priors=['learnable'], 
                # filter_priors=['only_learnable'], 
                train_first_stage=True, **kwargs):
        super().__init__(**kwargs)
        self.train_first_stage = train_first_stage
        if self.planes == 64 and not self.train_first_stage:
            pass
        else:
            self.filter_priors = filter_priors
            conv_cfg = kwargs['conv_cfg']
            planes = kwargs['planes']
            dilation = kwargs['dilation']

            self.mean = nn.AvgPool2d(kernel_size=3, stride=self.conv2_stride, padding=1)
            self.llpf = LLPFConv(channels=planes, stride=self.conv2_stride, padding=1)
            self.learnable_conv1 = build_conv_layer(
                    conv_cfg,
                    planes,
                    planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=dilation,
                    bias=False)

    def forward(self, x):
        """Forward function."""
        if self.planes == 64 and not self.train_first_stage:
            # print(self)
            return super().forward(x)

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            direct_out = self.conv2(out)
            if 'mean' in self.filter_priors:
                mean_out = self.mean_conv1(self.mean(out))
                # mean_out =self.mean(self.mean_conv1(out))
                direct_out += mean_out
                # print('mean')
            if 'gaussian' in self.filter_priors:
                gaussian_out = self.gaussian_conv1(self.gaussian(out))
                # gaussian_out = self.gaussian(self.gaussian_conv1(out))
                direct_out += gaussian_out
                # print('gaussian')
            if 'laplacian' in self.filter_priors:
                laplacian_out = self.laplacian(self.laplacian_dwconv1(self.laplacian_conv1(out)))
                direct_out += laplacian_out
                # print('laplacian')
            if 'learnable' in self.filter_priors:
                learnable = self.learnable_conv1(self.llpf(out))
                # learnable =self.llpf(self.learnable_conv1(out))
                direct_out += learnable
                # print('learnable')
            if 'only_learnable' in self.filter_priors:
                learnable = self.learnable_conv1(self.llpf(out))
                # learnable =self.llpf(self.learnable_conv1(out))
                direct_out = learnable
                # print('only_learnable')
            out = direct_out

            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# ============================== 复刻 fork 基类改动：冻结 stage 用标准 Bottleneck，未冻结 stage 用自定义 block

class _ResNetSmoothBase(ResNet):
    """与 fork 的 ResNet 基类一致：__init__ 构建 stage 时，冻结 stage（i < frozen_stages）
    使用标准 Bottleneck，未冻结 stage 使用 self.block（SmoothPriorBottleneck，即 SCB）。
    保证与发布权重的参数结构逐位一致。"""

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)
        # self._make_stem_layer(in_channels, stem_channels, caffe=style == 'caffe')

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block if i >= frozen_stages else Bottleneck, ### for smooth
                # block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
        # self.gaussian_filter = GaussianBlurConv(64, 2).cuda()
        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)



# ============================== ResNet 变体（AWD backbone）
@BACKBONES.register_module()
class ResNetAdaD(_ResNetSmoothBase):
    r"""
    Use a series avgpool to replace maxpool for downsample
    """
    # arch_settings = {
    #     18: (BasicBlock, (2, 2, 2, 2)),
    #     34: (BasicBlock, (3, 4, 6, 3)),
    #     50: (AdaBottleneck, (3, 4, 6, 3)),
    #     101: (AdaBottleneck, (3, 4, 23, 3)),
    #     152: (AdaBottleneck, (3, 8, 36, 3))
    # }
    def __init__(self, **kwargs):
        super(ResNetAdaD, self).__init__(strides=(1, 1, 1, 1), **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool= nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.mean_filter = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        self.AdaD = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ),
            AdaDConv(in_channels=256, kernel_size=3, stride=2, groups=256),
            AdaDConv(in_channels=512, kernel_size=3, stride=2, groups=512),
            AdaDConv(in_channels=1024, kernel_size=3, stride=2, groups=1024),
        ])


    def forward(self, x, denoise=True):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = self.AdaD[i](x)
            x = res_layer(x)
            if i in self.out_indices:
                # print(i)
                outs.append(x)
        return tuple(outs)

@BACKBONES.register_module()
class ResNetAdaDSmoothPrior(_ResNetSmoothBase):
    r"""
    Use a series avgpool to replace maxpool for downsample
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (SmoothPriorBottleneck, (3, 4, 6, 3)),
        101: (SmoothPriorBottleneck, (3, 4, 23, 3)),
        152: (SmoothPriorBottleneck, (3, 8, 36, 3))
    }
    def __init__(self, **kwargs):
        if kwargs['style'] == 'pytorch':
            strides = (1, 2, 2, 2)
        else:
            strides=(1, 1, 1, 1)
        super().__init__(strides=strides, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool= nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.mean_filter = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        if kwargs['style'] == 'pytorch':
            self.AdaD = nn.ModuleList([
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    # nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
                    # nn.MaxPool2d(kernel_size=2, stride=1),
                    # AdaDConv(in_channels=64, kernel_size=3, stride=2, groups=64),
                ),
                AdaDConv(in_channels=256, kernel_size=3, stride=1, groups=256),
                AdaDConv(in_channels=512, kernel_size=3, stride=1, groups=512),
                AdaDConv(in_channels=1024, kernel_size=3, stride=1, groups=1024),
            ])
        else:
            self.AdaD = nn.ModuleList([
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    # nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
                    # nn.MaxPool2d(kernel_size=2, stride=1),
                    # AdaDConv(in_channels=64, kernel_size=3, stride=2, groups=64),
                ),
                AdaDConv(in_channels=256, kernel_size=3, stride=2, groups=256),
                AdaDConv(in_channels=512, kernel_size=3, stride=2, groups=512),
                AdaDConv(in_channels=1024, kernel_size=3, stride=2, groups=1024),
            ])
    def forward(self, x):
        """Forward function."""
        # x = self.gfilter(x)
        # x = BilateralFilter(x, ksize=3, stride=1, sigmaSpace=1.0)
        # x = self.AdaD[0](x)
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = self.AdaD[i](x)
            x = res_layer(x)

            if i in self.out_indices:
                # print(i)
                outs.append(x)
            if i != len(self.res_layers) - 1:
                pass
        return tuple(outs)


# ============================== FPN 增强版（forward 支持 return_backbone，覆盖注册）

class FPNSmooth(_OfficialFPN):
    """fork 版 FPN：forward 额外返回 backbone 输入特征（DSL 特征一致性损失需要）。
    return_backbone=False 时与官方 FPN 行为完全一致。"""

    @auto_fp16()
    def forward(self, inputs, return_backbone=False):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        if return_backbone:
            return inputs, tuple(outs)
        else:
            return tuple(outs)


NECKS.register_module('FPN', force=True, module=FPNSmooth)



# ============================== DSL 检测器: from detectors/mask_rcnn.py
@DETECTORS.register_module()
class MaskRCNNNoiseInv(MaskRCNN):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self, **args):
        super().__init__(**args)
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      return_proposal=False,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        backbone_x, x = self.extract_feat(img, return_backbone=True)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        if return_proposal:
            return losses, backbone_x, x, proposal_list
        else:
            return losses, backbone_x, x

    def copy_self(self, load_from):
        # return None
        # print('copy')
        raise Exception
        self.cp = copy.deepcopy(self)
        teacher_ckpt = load_from
        print('load teacher param from:', teacher_ckpt)
        load_checkpoint(self.cp, teacher_ckpt, map_location='cuda')

    # @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            losses = dict()
            noisy_img = kwargs.pop('noisy_img')
            clean_losses, clean_backbone_x, clean_x = self.forward_train(img, img_metas, **kwargs)
            noisy_losses, noisy_backbone_x, noisy_x = self.forward_train(noisy_img, img_metas, **kwargs)
            losses['noise_inv_loss'] = 0
            for idx, (_noisy, _clean) in enumerate(zip(noisy_backbone_x[:], clean_backbone_x[:])):
                losses['noise_inv_loss'] += 0.01 * torch.clamp(F.mse_loss(input=_noisy, target=_clean, reduction='none'), max=0.1, min=0).mean()
            for k in clean_losses: losses[f'clean_{k}'] = clean_losses[k]
            for k in noisy_losses: losses[f'{k}'] = noisy_losses[k]
            for k in losses: 
                if isinstance(losses[k], int): 
                    losses.pop(k)
            return losses
        else:
            # return self.forward_test(img, img_metas, **kwargs)
            return super().forward_test(img, img_metas, **kwargs)

    def extract_feat(self, img, return_backbone=True):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            if return_backbone:
                backbone_x, fpn_x = self.neck(x, return_backbone)
                return backbone_x, fpn_x
            else:
                fpn_x = self.neck(x, return_backbone)
                return fpn_x
        else:
            return x
        

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        # img_residual = F.interpolate(self.unet(img), (img.size(-2), img.size(-1)))
        # img = img + img_residual
        x = self.extract_feat(img, return_backbone=False)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        def feature2image(feat, save_dir, name, normalize=False):
            _, _, h, w = feat.shape
            feat = feat.mean(dim=1).repeat(3, 1, 1)
            feat = F.interpolate(feat[None, ], size=(400, 600), mode='bilinear')[0]
            save_img = feat.permute(1, 2, 0).cpu().numpy()
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

        # img = self.unet(img) + img
        backbone_x, x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)


# ============================== RAW 低光噪声合成：unprocess / process / 直方图匹配
def random_ccm(): # 创建随机颜色校正矩阵作为Device RGB和sRGB之间的转换
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181]]]
    num_ccms = len(xyz2cams)
    xyz2cams = np.array(xyz2cams)
    weights = np.random.uniform(1e-8, 1e8, size=(num_ccms, 1, 1))
    #  weights = np.ones((num_ccms, 1, 1))
    weights_sum = np.sum(weights, axis=0)
    xyz2cam = np.sum(xyz2cams * weights, axis=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / np.sum(rgb2cam, axis=-1, keepdims=True)
    return rgb2cam

def get_calibrated_cam2rgb():
    cam2rgb_matrix = np.array([[ 2.04840695, -1.27161572,  0.22320878],
                               [-0.22163155,  1.77694640, -0.55531485],
                               [-0.00770995, -0.59257895,  1.60028890]], dtype=np.float32)
    return cam2rgb_matrix

def random_gains():# 创建随机白平衡参数
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / np.random.normal(0.8, 0.1) 

    # Red and blue gains represent white balance.
    red_gain = np.random.uniform(1.9, 2.4)
    blue_gain = np.random.uniform(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain

def inverse_smoothstep(image): # 逆色调映射
    """Approximately inverts a global tone mapping curve."""
    image = np.clip(image, 0.0, 1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * image) / 3.0) # 全局逆色调映射仿真函数

def gamma_expansion(image): # gamma扩展，将non-linear值转换为linear
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return np.maximum(image, 1e-8) ** 2.2

def apply_ccm(image, ccm): # 应用颜色校正矩阵
    """Applies a color correction matrix."""
    shape = image.shape
    image = np.reshape(image, [-1, 3])
    image = np.tensordot(image, ccm, [[-1], [-1]])
    return np.reshape(image, shape)

def safe_invert_gains(image, rgb_gain, red_gain, blue_gain): # 逆白平衡同时处理过饱和像素
    """Inverts gains while safely handling saturated pixels."""
    gains = np.stack((1.0 / red_gain, 1.0, 1.0 / blue_gain)) / rgb_gain
    gains = gains.squeeze()
    gains = gains[None, None, :]
    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = np.mean(image, axis=-1, keepdims=True)
    inflection = 0.9
    mask = (np.maximum(gray - inflection, 0.0) / (1.0 - inflection)) ** 2.0
    safe_gains = np.maximum(mask + (1.0 - mask) * gains, gains)
    return image * safe_gains

def mosaic(image): # 将三通道RGB转换成四通道Raw,此处为RGGB bayer
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape
    red = image[0::2, 0::2, 0]
    green_red = image[0::2, 1::2, 1]
    green_blue = image[1::2, 0::2, 1]
    blue = image[1::2, 1::2, 2]
    # out = np.stack((red, green_red, green_blue, blue), axis=-1) # RGGB 
    out = np.stack((red, green_red, blue, green_blue), axis=-1) # RGBG 
    out = np.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
    return out

def unprocess(image):
    """Unprocesses an image from sRGB to realistic raw data."""

    # 为unprocessing过程创建必须的随机矩阵和随机数
    # Randomly creates image metadata.
    # rgb2cam = random_ccm()
    # cam2rgb = np.linalg.inv(rgb2cam)
    cam2rgb = get_calibrated_cam2rgb()
    rgb2cam = np.linalg.inv(cam2rgb)
    rgb_gain, red_gain, blue_gain = random_gains()

    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image) # 逆色调映射
    # Inverts gamma compression.
    image = gamma_expansion(image) # 线性化
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam) # 逆颜色校正
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain) # 逆白平衡
    # Clips saturated pixels.
    image = np.clip(image, 0.0, 1.0)
    # Applies a Bayer mosaic.
    image = mosaic(image)

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return image, metadata


def unpack_raw_bayer(img): # 将四通道Raw转换成单通道bayer阵列
    # unpack 4 channels to Bayer image
    img4c = np.transpose(img, (2, 0, 1))
    _, h, w = img4c.shape

    H = int(h * 2)
    W = int(w * 2)

    cfa_img = np.zeros((H, W), dtype=np.float32)

    cfa_img[0:H:2, 0:W:2] = img4c[0, :,:]
    cfa_img[0:H:2, 1:W:2] = img4c[1, :,:]
    cfa_img[1:H:2, 1:W:2] = img4c[2, :,:]
    cfa_img[1:H:2, 0:W:2] = img4c[3, :,:]
    
    return cfa_img

def process(img, shape):
    bayer = (unpack_raw_bayer(img) * 255.).round().astype(np.uint8) # 将四通道Raw转换成单通道bayer阵列，并乘以255.存为图像数据格式
    RGB = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2RGB) # 将bayer阵列转换成三通道RGB（无白平衡和颜色校正）
    return cv2.resize(RGB, shape) # 将图像resize成指定尺寸





# ============================== 噪声模型与 pipeline transform: from pipelines/noisemodel/dark_noising.py
def calculate_brightness(img):
    YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return np.mean(YUV[:, :, 0])

def adjust_random_brightness(img, mean=9.977646797874655, std=8.723007766522453):
    est_brightness = calculate_brightness(img)
    rand_brightness = np.maximum(1e-3, np.random.normal(mean, std))
    ratio = rand_brightness / est_brightness
    return img * ratio

def add_noise(image, noise_type):
    if noise_type is None:
        return image
    if noise_type == 'gaussian':
        # 初始化noisemodel，噪声类型为'g'，标定相机为CanonEOS5D4
        noisemodel = NoiseModel(model='g', camera='CanonEOS5D4')
    elif noise_type == 'gaussian-poisson':
        # 初始化noisemodel，噪声类型为'pg'，标定相机为CanonEOS5D4
        noisemodel = NoiseModel(model='pg', camera='CanonEOS5D4')
    elif noise_type == 'physics-based':
        # 初始化noisemodel，噪声类型为'PGRU'，标定相机为CanonEOS5D4
        noisemodel = NoiseModel(model='PGRU', camera='CanonEOS5D4')
    noisy_image = noisemodel(image)  # 添加噪声
    noisy_image = np.clip(noisy_image, 0, 1)  # 取值限定0~1
    return noisy_image

def adjust_random_brightness(image, s_range=(0.1, 0.3)):  # 在s_range范围内设置随机亮度
    assert s_range[0] < s_range[1]
    ratio = np.random.rand() * (s_range[1] - s_range[0]) + s_range[0]
    return image * ratio

def add_gaussian_noise(image, mean=0, std=0.25):
    noise = np.random.normal(mean, std, image.shape)
    return image + noise

def random_noise_levels():  # 初始化rand and shot noise模型参数随机
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    def line(x): return 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(0, 0.26)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise

def add_read_and_shot_noise(image, shot_noise=0.01, read_noise=0.005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    noise = np.random.normal(0, np.sqrt(variance), size=(variance.shape))
    return image + noise


@PIPELINES.register_module()
class NoiseModel:
    def __init__(self, param_dir=None, model='g', camera='CanonEOS5D4', cfa='bayer', 
                use_255=True, pre_adjust_brightness=False, mode='addnoise', to_rgb=True, fix_noise=False,
                dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)):
        super().__init__()
        modes = ['addnoise', 'unprocess_addnoise', 'unprocess']
        assert mode in modes
        self.mode = mode
        self.to_rgb = to_rgb
        self.fix_noise = fix_noise
        # if self.fix_noise:
            # self.fix_params = self._sample_params()

        self.camera = camera
        if param_dir is None:
            try:
                self.param_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'camera_params')
            except:
                print('please specify the location of camera parameters, e.g., ~/code/mmdetection/mmdet/datasets/pipelines/noisemodel/camera_params')
                raise Exception
        else:
            self.param_dir = param_dir

        print('[i] NoiseModel with {}'.format(self.param_dir))
        print('[i] camera: {}'.format(self.camera))
        print('[i] using noise model {}'.format(model))
        print('[i] dark ratio {}'.format(dark_ratio))
        print('[i] noise ratio {}'.format(noise_ratio))

        self.camera_params = {}
        self.camera_params[camera] = np.load(join(
            self.param_dir, camera+'_params.npy'), allow_pickle=True).item()  # 加载指定相机标定参数

        self.model = model
        self.use_255 = use_255
        self.pre_adjust_brightness = pre_adjust_brightness

        self.dark_ratio = dark_ratio
        self.noise_ratio = noise_ratio

    def _sample_params(self):
        # if self.fix_noise:
            # return self.fix_params
        camera = self.camera
        Q_step = 1  # 量化层级

        profiles = ['Profile-1']
        saturation_level = 16383 - 512  # 相机标定值，白点减去黑电平值

        # 调取相机标定参数
        camera_params = self.camera_params[camera]
        Kmin = camera_params['Kmin']
        Kmax = camera_params['Kmax']

        # 抽取标定参数中的随机数
        G_shape = np.random.choice(camera_params['G_shape'])
        ind = np.random.randint(0, camera_params['color_bias'].shape[0])
        color_bias = camera_params['color_bias'][ind, :]
        profile = np.random.choice(profiles)
        camera_params = camera_params[profile]

        # 初始化noisemodel模型中的随机数
        log_K = np.random.uniform(low=np.log(Kmin), high=np.log(Kmax))
        log_g_scale = np.random.standard_normal() * camera_params['g_scale']['sigma'] * 1 +\
            camera_params['g_scale']['slope'] * \
            log_K + camera_params['g_scale']['bias']
        log_G_scale = np.random.standard_normal() * camera_params['G_scale']['sigma'] * 1 +\
            camera_params['G_scale']['slope'] * \
            log_K + camera_params['G_scale']['bias']
        log_R_scale = np.random.standard_normal() * camera_params['R_scale']['sigma'] * 1 +\
            camera_params['R_scale']['slope'] * \
            log_K + camera_params['R_scale']['bias']

        # 计算noisemodel中的参数
        K = np.exp(log_K)
        g_scale = np.exp(log_g_scale)
        G_scale = np.exp(log_G_scale)
        R_scale = np.exp(log_R_scale)

        # radio控制噪声的大小，ratio取值越大噪声越明显
        ratio = np.random.uniform(low=self.noise_ratio[0], high=self.noise_ratio[1])
        # ratio = np.random.uniform(low=100, high=300)
        # ratio = np.random.uniform(low=20, high=100)
        # ratio = np.random.uniform(low=1, high=300)
        # ratio = 1
        # ratio = np.random.uniform(low=20, high=50)
        # , dtype=np.float32
        # print(K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio)
        return np.array([K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio])

    def __call__(self, results, params=None):
        _dark = np.random.uniform(self.dark_ratio[0], self.dark_ratio[1])
        for key in results.get('img_fields', ['img']):
            if self.mode == 'addnoise':
                if self.use_255:
                    results[key] = self.add_noise_255(results[key], params) * _dark
                else:
                    results[key] = self.add_noise(results[key], params) * _dark
            elif self.mode == 'unprocess':
                results[key] = self.unprocess(results[key], addnoise=False) * _dark
            elif self.mode == 'unprocess_addnoise':
                results[key] = self.unprocess(results[key], addnoise=True) * _dark
            else:
                raise NotImplementedError
                
        return results

    def add_noise_255(self, y, params=None):
        # print(y, y.shape)
        y = (1.0 / 255.) * np.array(y).astype(np.float32)
        z = self.add_noise(y, params)
        z = (z * 255).round() # .astype(np.uint8)
        # z = (z * 255) # .astype(np.uint8)
        # Image.fromarray(z.astype(np.uint8)).show()
        # exit()
        return z

    def add_noise(self, y, params=None):
        if params is None:
            K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio = self._sample_params()
        else:
            K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio = params

        y = y * saturation_level
        y = y / ratio

        # noisemodel中的photon shot noise
        if 'P' in self.model:
            z = np.random.poisson(y / K).astype(np.float32) * K
        elif 'p' in self.model:
            z = y + np.random.randn(*y.shape).astype(np.float32) * \
                np.sqrt(np.maximum(K * y, 1e-10))
        else:
            z = y

        # noisemodel中的read noise
        if 'g' in self.model:
            z = z + np.random.randn(*y.shape).astype(np.float32) * \
                np.maximum(g_scale, 1e-10)  # Gaussian noise
        elif 'G' in self.model:
            z = z + stats.tukeylambda.rvs(G_shape, loc=0, scale=G_scale,
                                          size=y.shape).astype(np.float32)  # Tukey Lambda

        # noisemodel会议版本未提及此噪声, 根据代码是根据相机标定参数中的随机数随机添加色偏
        if 'B' in self.model:
            z = self.add_color_bias(z, color_bias=color_bias)

        # noisemodel中的row noise
        if 'R' in self.model:
            z = self.add_banding_noise(z, scale=R_scale)

        # noisemodel中的quantization noise
        if 'U' in self.model:
            z = z + np.random.uniform(low=-0.5*Q_step, high=0.5*Q_step)

        z = z * ratio
        z = z / saturation_level

        # post
        z = np.clip(z, 0, 1)
        # do not adjust brightness
        # z = adjust_random_brightness(z)
        return z

    def add_color_bias(self, img, color_bias):  # 添加随机色偏
        channel = img.shape[2]
        img = img + color_bias.reshape((1, 1, channel))
        return img

    def add_banding_noise(self, img, scale):  # 添加banding噪声，即论文中row noise
        channel = img.shape[2]
        img = img + \
            np.random.randn(img.shape[0], 1, channel).astype(np.float32) * scale
        return img

    def unprocess(self, image, addnoise=True, return_clean=False):
        # check the RGB or BGR
        # if image.mode != 'RGB':
            # image = image.convert("RGB")
        '''
        image: BGR
        '''
        if self.to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print('image', image.shape)
        # W, H = image.size
        H, W, _ = image.shape

        # resize for mosaic in unprocessing
        # image = image.resize((W // 2 * 2, H // 2 * 2))  # mosaic需要图像尺寸为偶数
        image = cv2.resize(image, (W // 2 * 2, H // 2 * 2))

        # print(image.shape)

        if self.use_255:
            image = np.array(image).astype(np.float32) / 255.  # 像素值归一化

        raw, metadata = unprocess(image)  # unprocess成四通道raw，按RGBG排列, -> 0~1
        # Image.fromarray(raw).show()
        # print(raw.shape)

        if addnoise :
            # self.add_noise 会归一化
            noisy_raw = self.add_noise(raw.copy())  # 添加噪声
            # noisy_raw = np.clip(noisy_raw, 0, 1)  # 取值限定0~1
            # dark_raw = adjust_random_brightness(noisy_raw, s_range=(0.2, 0.4)) # 随机调整pixel强度值在s_range范围内
            dark_raw = noisy_raw

            noisy_result = process(dark_raw, (W, H))
            if self.to_rgb:
                noisy_result = cv2.cvtColor(noisy_result, cv2.COLOR_RGB2BGR)

            if return_clean:
                result = process(raw, (W, H))
                if self.to_rgb:
                    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                return noisy_result, result
            else:
                return noisy_result
        else:
            # dark_raw = adjust_random_brightness(raw, s_range=(0.2, 0.4)) # 随机调整pixel强度值在s_range范围内
            dark_raw = raw
            # 0~255
            result = process(dark_raw, (W, H))  # 将四通道raw转换成三通道RGB（不进行白平衡和颜色校正），并resize保持与输入一致
            if self.to_rgb:
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            # Image.fromarray(result).show()
            # exit()
            return result

@PIPELINES.register_module()
class AddNoisyImg(NoiseModel):
    def __init__(self, param_dir=None, model='g', camera='CanonEOS5D4', cfa='bayer', 
                use_255=True, pre_adjust_brightness=False, mode='addnoise', fix_noise=False,
                dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)):
        super().__init__(param_dir=param_dir, model=model, camera=camera, cfa=cfa, 
                         use_255=use_255, pre_adjust_brightness=pre_adjust_brightness, mode=mode, fix_noise=fix_noise,
                         dark_ratio=dark_ratio, noise_ratio=noise_ratio)

    def __call__(self, results, params=None):
        results['ori_img'] = results['img'].copy()
        _dark = np.random.uniform(self.dark_ratio[0], self.dark_ratio[1])
        for key in results.get('img_fields', ['img']):
            if self.mode == 'addnoise':
                if self.use_255:
                    results['noisy_img'] = self.add_noise_255(results[key], params) * _dark
                else:
                    results['noisy_img'] = self.add_noise(results[key], params) * _dark
                
            # elif self.mode == 'unprocess':
                # results['img'] = self.unprocess(results[key], addnoise=False)
                # results['img_fields'].append('noisy_img')

            elif self.mode == 'unprocess_addnoise':

                results['noisy_img'], results['img'] = self.unprocess(results[key], addnoise=True, return_clean=True)
                results['noisy_img'] = results['noisy_img'] * _dark
                results['img'] = results['img'] * _dark
                # results['noisy_img'] = self.unprocess(results[key], addnoise=True) * _dark
                # results['img'] = self.unprocess(results[key], addnoise=False) * _dark
            else:
                raise NotImplementedError

        results['img_fields'].append('noisy_img')
        results['img_fields'].append('ori_img')
        return results

__all__ = [
    'AdaDConv', 'GaussianBlurConv', 'LaplacianConv', 'LLPFConv',
    'BaseBackbone', 'LayerNorm2d', 'ConvNeXtBlock', 'SmoothPriorConvNeXtBlock',
    'ConvNeXt', 'ConvNeXtAdaD', 'ConvNeXtAdaDSmoothPrior',
    'BasicBlock', 'SmoothPriorBottleneck', '_ResNetSmoothBase', 'ResNetAdaD',
    'ResNetAdaDSmoothPrior', 'FPNSmooth', 'MaskRCNNNoiseInv',
    'NoiseModel', 'AddNoisyImg', 'hist_match',
]

if __name__ == '__main__':
    print('[lis_pluggable] all custom components registered OK')
    print('[lis_pluggable] registered:', ', '.join(__all__))
