# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 01:14:23 2021

@author: chxy
"""

import cv2
import numpy as np

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

