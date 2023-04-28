# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 00:12:03 2021

@author: chxy
"""
from math import e
import cv2
import numpy as np
from PIL import Image
import scipy.stats as stats
from os.path import join
import random
from ...builder import PIPELINES
from numpy.testing._private.utils import print_assert_equal
from .process import process
from .unprocess import unprocess
import os.path as osp

def calculate_brightness(img):
    YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return np.mean(YUV[:, :, 0])


# 在以mean和std的高斯分布下调整亮度
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


@PIPELINES.register_module()
class LoadDarkPair:
    def __init__(self, img_dir, ext='png'):
        self.img_dir = img_dir
        self.ext = ext
        pass

    def __call__(self, results):
        file_name = results['ori_filename']
        img_id = int(osp.splitext(osp.split(file_name)[1])[0])
        dark_img_id = img_id + 1
        dark_img_path = osp.join(self.img_dir, f'JPEGImages/{dark_img_id}.{self.ext}')
        # print(dark_img_path)
        results['noisy_img'] = cv2.imread(dark_img_path)
        results['img_fields'].append('noisy_img')
        # for k in results: print(k, results[k])
        return results

@PIPELINES.register_module()
class Darker:
    def __init__(self, dark_ratio=(1.0, 1.0)) -> None:
        self.dark_ratio = dark_ratio
        
    def __call__(self, results):
        _dark = np.random.uniform(self.dark_ratio[0], self.dark_ratio[1])
        for key in results.get('img_fields', ['img']):
            results[key] = results[key] * _dark
        return results

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
                self.param_dir = '~/code/mmdetection/mmdet/datasets/pipelines/noisemodel/camera_params'
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

        # for key in results.get('img_fields', ['img']):
        #     if self.use_255:
        #         results['noisy_img'] = self.add_noise_255(results[key], params)
        #     else:
        #         results['noisy_img'] = self.add_noise(results[key], params)
        # # print(results.keys())
        # results['img_fields'].append('noisy_img')
        # # print(results['img'].shape, results['noisy_img'].shape)
        # return results

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


# 添加read and shot noise
def add_read_and_shot_noise(image, shot_noise=0.01, read_noise=0.005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    noise = np.random.normal(0, np.sqrt(variance), size=(variance.shape))
    return image + noise
