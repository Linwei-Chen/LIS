# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 00:30:15 2021

@author: chxy
"""

import os
import os.path as osp
import sys
sys.path.append('/home/ubuntu/code/mmdetection/mmdet/datasets/pipelines/noisemodel/')

import numpy as np
from PIL import Image

from process import process
from unprocess import unprocess
from dark_noising import *


def main(input_folder, output_folder, noise_type):
    if not osp.exists(output_folder):
        os.mkdir(output_folder)

    if noise_type == 'gaussian':
        noisemodel = NoiseModel(model='g', camera='CanonEOS5D4')  # 初始化noisemodel，噪声类型为'g'，标定相机为CanonEOS5D4
    elif noise_type == 'gaussian-poisson':
        noisemodel = NoiseModel(model='pg', camera='CanonEOS5D4')  # 初始化noisemodel，噪声类型为'pg'，标定相机为CanonEOS5D4
    elif noise_type == 'physics-based':
        noisemodel = NoiseModel(model='PGBRU', camera='CanonEOS5D4')  # 初始化noisemodel，噪声类型为'PGBRU'，标定相机为CanonEOS5D4

    for filename in sorted(os.listdir(input_folder)):
        image = Image.open(osp.join(input_folder, filename))  # 读取图像
        if image.mode != 'RGB':
            image = image.convert("RGB")
        W, H = image.size

        # resize for mosaic in unprocessing
        image = image.resize((W // 2 * 2, H // 2 * 2))  # mosaic需要图像尺寸为偶数

        image = np.array(image).astype(np.float32) / 255.  # 像素值归一化
        raw, metadata = unprocess(image)  # unprocess成四通道raw，按RGBG排列

        if noise_type is None:
            # dark_raw = adjust_random_brightness(raw, s_range=(0.2, 0.4)) # 随机调整pixel强度值在s_range范围内
            dark_raw = raw

        else:
            noisy_raw = noisemodel(raw)  # 添加噪声
            noisy_raw = np.clip(noisy_raw, 0, 1)  # 取值限定0~1
            # dark_raw = adjust_random_brightness(noisy_raw, s_range=(0.2, 0.4)) # 随机调整pixel强度值在s_range范围内
            dark_raw = noisy_raw

        result = process(dark_raw, (W, H))  # 将四通道raw转换成三通道RGB（不进行白平衡和颜色校正），并resize保持与输入一致
        result = Image.fromarray(result)
        result.save(osp.join(output_folder, filename[:-4] + '.png'))  # 保存图像


if __name__ == "__main__":
    main(input_folder=r'/home/ubuntu/2TB/dataset/VOCdevkit/VOC2012/JPEGImages',  # 输入文件夹路径（COCO、VOC原图）
         output_folder=r'/home/ubuntu/2TB/dataset/VOCdevkit/VOC2012/Gaussian',
         # 输出文件夹路径：unp_None, unp_gaussian, unp_gaussian-poisson, unp_physics-based
         noise_type='gaussian')  # 可选噪声类型：None, 'gaussian', 'gaussian-poisson', 'physics-based'
