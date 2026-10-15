# -*- coding: utf-8 -*-
"""将 SCB 的 LLPF 低通先验分支（softmax 加权的 3x3 depthwise 核）重参数化合并进
相邻卷积（conv2 + learnable_conv1），得到推理零开销模型（论文 Sec. 推理加速）。

用法: python tools/reparameter.py IN_CKPT OUT_CKPT
"""
import sys

import torch

model = torch.load(sys.argv[1], map_location='cpu')
    
for k in model['state_dict']: 
    if '.conv2.' in k:
        print(k)
        # llpf = model['state_dict']['backbone.layer2.0.llpf.weight']
        try:
            llpf_conv1 = model['state_dict'][k.replace('conv2', 'learnable_conv1')]
            llpf = model['state_dict'][k.replace('conv2', 'llpf')]
            llpf_conv1 = llpf_conv1.repeat(1, 1, 3, 3)
            llpf = llpf.reshape(-1, 1, 9).softmax(dim=-1).reshape(-1, 1, 3, 3)
            model['state_dict'][k] += llpf_conv1 * llpf
            # print(k)
        except Exception:
            pass

torch.save(model, sys.argv[2])