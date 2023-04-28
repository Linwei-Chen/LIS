# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 23:32:03 2021

@author: chxy
"""

import numpy as np
import scipy.stats as stats
from os.path import join
import random

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