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
from tqdm import tqdm
from ISP import raw2rgb, raw2rawrgb

def extract_exposure(raw_path): # 使用exifread库读取RAW文件元数据并提取曝光时间
    raw_file = open(raw_path, 'rb')
    exif_file = exifread.process_file(raw_file, details=False, strict=True)

    if 'EXIF ExposureTime' in exif_file:
        exposure_str = exif_file['EXIF ExposureTime'].printable
    else:
        exposure_str = exif_file['Image ExposureTime'].printable
    if '/' in exposure_str:
        fenmu = float(exposure_str.split('/')[0])
        fenzi = float(exposure_str.split('/')[-1])
        exposure = fenmu / fenzi
    else:
        exposure = float(exposure_str)
    return exposure

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
    # print(out)
    return out

def postprocess_bayer(rawpath, img4c):    
    img4c = np.clip(img4c, 0, 1)

    #unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    G2 = np.where(raw_pattern==3)
    B = np.where(raw_pattern==2)
    
    white_point = raw.white_level
    black_level = np.array(raw.black_level_per_channel)[:,None,None]

    img4c = img4c * (white_point - black_level) + black_level
    
    img_shape = raw.raw_image_visible.shape
    H = img_shape[0]
    W = img_shape[1]

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :,:]
    raw.raw_image_visible[G1[0][0]:H:2,G1[1][0]:W:2] = img4c[1, :,:]
    raw.raw_image_visible[B[0][0]:H:2,B[1][0]:W:2] = img4c[2, :,:]
    raw.raw_image_visible[G2[0][0]:H:2,G2[1][0]:W:2] = img4c[3, :,:]

    out = raw.postprocess(use_camera_wb=False, user_wb=[1,1,1,1], half_size=False, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1.96875, 1, 1.444, 1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)    
    # out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None, gamma=(2.2, 0))

    # out = raw.postprocess(user_wb=[1,1,1,1], half_size=True, output_bps=8, no_auto_bright=True)
    return out


# 修改输入文件夹路径，且命名数字符合dark = normal + 1 (eg: normal: 1.CR2, dark: 2.CR2)
normal_input_folder = '/home/ubuntu/2TB/dataset/LOD/all_rename'
dark_input_folder = '/home/ubuntu/2TB/dataset/LOD/all_rename'

# normal和dark RGB保存路径
normal_output_folder = '/home/ubuntu/2TB/dataset/LOD/New_RAW_Normal'
dark_output_folder = '/home/ubuntu/2TB/dataset/LOD/New_RAW_Dark'

if not osp.exists(normal_output_folder):
    os.mkdir(normal_output_folder)
if not osp.exists(dark_output_folder):
    os.mkdir(dark_output_folder)

normal_img_list = sorted(os.listdir(normal_input_folder))
normal_img_list = [i for i in normal_img_list if 'CR2' in i]
# print(normal_img_list)
# exit()

# for filename in tqdm(sorted(os.listdir(normal_input_folder))):
for filename in tqdm(normal_img_list):
    # if osp.splitext(filename)[0]
    if filename[-3:] == 'CR2' and int(filename[:-4]) % 2 == 1: # 判断文件是否是RAW格式
        # prefix = filename[:4] # 前缀，eg：7U6A
        # number = filename[4:-4] # 中间数字：eg：2714
        # suffix = filename[-4:] # 后置，eg：.CR2
        # print(prefix, number, suffix)
        # dark_name = prefix + str(int(number) + 1) + suffix # 对应dark图片名称
        dark_name = str(int(filename[:-4]) + 1) + '.CR2'
        # print(dark_name)
        if osp.exists(osp.join(dark_input_folder, dark_name)): # 若存在对应dark图片
            # normal image processing
            
            normal_raw = rawpy.imread(osp.join(normal_input_folder, filename)) # 读取normal RAW
            normal_img4c = pack_raw_bayer(normal_raw) # RAW文件解码成四通道
            # (4, 2251, 3372)
            # normal_img = raw2rgb(normal_img4c, normal_raw).transpose((1, 2, 0))
            normal_img = raw2rawrgb(normal_img4c, normal_raw).transpose((1, 2, 0))
            normal_img = (normal_img * 255).round().astype(np.uint8)
            # normal_img = postprocess_bayer(osp.join(normal_input_folder, filename), normal_img4c) # 四通道bayer转三通道
            
            # normal_img = cv2.resize(normal_img, (6720, 4480)) # 将图像resize成原图尺寸, 防止内部计算里面牵扯到的一些不能整除的问题而产生的size的微小变化
            normal_img = cv2.resize(normal_img, (1200, 800))
            normal_img = Image.fromarray(normal_img) # 图像转成Image格式
            normal_img.save(osp.join(normal_output_folder, filename[:-4]+'.png')) # 保存normal图像，存储为png为无损存储，jpg格式会进行压缩
            
            # exit()
            continue
            
            normal_exposure = extract_exposure(osp.join(normal_input_folder, filename)) # normal图片曝光时间
            dark_exposure = extract_exposure(osp.join(dark_input_folder, dark_name)) # dark图片曝光时间
            # print(f'normal_exposure:{normal_exposure}', f'dark_exposure:{dark_exposure}')
            wb = np.array(normal_raw.camera_whitebalance) 
            # print(f'wb:{wb}')

            dark_raw = rawpy.imread(osp.join(dark_input_folder, dark_name)) # 读取dark RAW
            dark_img4c = pack_raw_bayer(dark_raw) / dark_exposure * normal_exposure # RAW文件解码成四通道并按曝光时间调整亮度
            # print(dark_img4c)
            
            # dark_img4c = pack_raw_bayer(dark_raw) # RAW文件解码成四通道并按曝光时间调整亮度

            # dark_img = raw2rgb(dark_img4c, normal_raw).transpose((1, 2, 0)) # to RGB
            dark_img = raw2rawrgb(dark_img4c).transpose((1, 2, 0)) # to RAW
            dark_img = (dark_img * 255)
            # print(dark_img)
            # dark_img = postprocess_bayer(osp.join(dark_input_folder, dark_name), dark_img4c) # 四通道bayer转三通道

            # dark_img = cv2.resize(dark_img, (6720, 4480)) # 将图像resize成原图尺寸, 防止内部计算里面牵扯到的一些不能整除的问题而产生的size的微小变化
            dark_img = cv2.resize(dark_img, (1200, 800), cv2.INTER_NEAREST)
            dark_img = Image.fromarray(dark_img.round().astype(np.uint8)) # 图像转成Image格式
            dark_img.save(osp.join(dark_output_folder, dark_name[:-4]+'.png')) # 保存dark图像，存储为png为无损存储，jpg格式会进行压缩
            # cv2.imwrite(osp.join(dark_output_folder, dark_name[:-4]+'.png'), dark_img * 255)
            # dark_img = cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR)
            # cv2.imencode('.png', dark_img.astype(np.float32))[1].tofile(osp.join(dark_output_folder, dark_name[:-4]+'.png'))

            exit()