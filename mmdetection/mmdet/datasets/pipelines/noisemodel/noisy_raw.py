# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:28:43 2021

@author: hy
"""

import os
import os.path as osp

import rawpy
import exifread

import numpy as np
from PIL import Image
import cv2

from ISP import raw2rgb
from dark_noising import *

def pack_raw_bayer(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

    white_point = raw.white_level
    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)
    # print(black_level[0,0,0], white_point)
    
    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out

def add_noise(image, noise_type='physics-based'): # 可选噪声类型：None, 'gaussian', 'gaussian-poisson', 'physics-based')
    if noise_type is None:
        return image
    if noise_type == 'gaussian':
        noisemodel = NoiseModel(model='g', camera='CanonEOS5D4') # 初始化noisemodel，噪声类型为'g'，标定相机为CanonEOS5D4
    elif noise_type == 'gaussian-poisson':
        noisemodel = NoiseModel(model='pg', camera='CanonEOS5D4') # 初始化noisemodel，噪声类型为'pg'，标定相机为CanonEOS5D4
    elif noise_type == 'physics-based':
        noisemodel = NoiseModel(model='PGRU', camera='CanonEOS5D4') # 初始化noisemodel，噪声类型为'PGBRU'，标定相机为CanonEOS5D4
    noisy_image = noisemodel(image) #添加噪声
    noisy_image = np.clip(noisy_image, 0, 1) # 取值限定0~1
    return noisy_image

normal_input_folder = r'H:\object_detection_data\noisy_test\Normal_raw'

# normal和noisy RGB保存路径
normal_output_folder = r'H:\object_detection_data\noisy_test\physics-based_normal'
noisy_output_folder = r'H:\object_detection_data\noisy_test\physics-based_noisy'

if not osp.exists(normal_output_folder):
    os.mkdir(normal_output_folder)
if not osp.exists(noisy_output_folder):
    os.mkdir(noisy_output_folder)

for filename in sorted(os.listdir(normal_input_folder)):
    if filename[-4:] == '.CR2':
        normal_raw = rawpy.imread(osp.join(normal_input_folder, filename)) # 读取normal RAW
        normal_img4c = pack_raw_bayer(normal_raw) # RAW文件解码成四通道

        noisy_img4c = add_noise(normal_img4c, noise_type='physics-based') # 可选噪声类型：None, 'gaussian', 'gaussian-poisson', 'physics-based')

        normal_img = raw2rgb(normal_img4c, normal_raw).transpose((1, 2, 0))
        normal_img = (normal_img * 255).round().astype(np.uint8)
        # normal_img = postprocess_bayer(osp.join(normal_input_folder, filename), normal_img4c) # 四通道bayer转三通道
        
        normal_img = cv2.resize(normal_img, (6720, 4480)) # 将图像resize成原图尺寸, 防止内部计算里面牵扯到的一些不能整除的问题而产生的size的微小变化
        normal_img = Image.fromarray(normal_img) # 图像转成Image格式
        normal_img.save(osp.join(normal_output_folder, filename[:-4]+'.png')) # 保存normal图像，存储为png为无损存储，jpg格式会进行压缩
                
        noisy_img = raw2rgb(noisy_img4c, normal_raw).transpose((1, 2, 0))
        noisy_img = (noisy_img * 255).round().astype(np.uint8)
        noisy_img = cv2.resize(noisy_img, (6720, 4480))
        noisy_img = Image.fromarray(noisy_img)
        noisy_img.save(osp.join(noisy_output_folder, filename[:-4]+'.png'))