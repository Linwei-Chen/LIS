import cv2
import numpy as np


def match_histograms(image, reference, multichannel=True):
    """按累积分布直方图匹配，把 reference 的灰度分布映射到 image 上。
    与 skimage 0.16 的 skimage.exposure.match_histograms 等价（BSD-3），内嵌以移除 skimage 依赖
    （新版 skimage 的 numpy 兼容性断言与本项目环境冲突）。"""
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')

    def _match_cumulative_cdf(source, template):
        src_values, src_unique_indices, src_counts = np.unique(
            source.ravel(), return_inverse=True, return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
        src_quantiles = np.cumsum(src_counts) / source.size
        tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        return interp_a_values[src_unique_indices].reshape(source.shape)

    if image.ndim == 2:
        return _match_cumulative_cdf(image, reference)

    if image.shape[-1] != reference.shape[-1]:
        raise ValueError('Image and reference must have the same number of channels.')

    matched = np.empty(image.shape, dtype=image.dtype)
    for channel in range(image.shape[-1]):
        matched[..., channel] = _match_cumulative_cdf(
            image[..., channel], reference[..., channel])
    return matched
from ...builder import PIPELINES
from glob import glob
import numpy as np



@PIPELINES.register_module()
class HistMatch:
    def __init__(self, img_dir, ratio=0.5, multichannel=True) -> None:
        _EXT = ['png', 'jpg', 'JPG']
        self.img_dir = img_dir
        self.img_path_list = glob(self.img_dir)
        self.ratio = ratio
        self.multichannel = multichannel

    def __call__(self, results):
        if np.random.uniform(0, 1.0) < self.ratio:
            while True:
                rand_img_idx = np.random.randint(low=0, high=len(self.img_path_list))
                rand_img = self.img_path_list[rand_img_idx]
                try:
                    ref = cv2.imread(rand_img)
                    break
                except Exception:
                    continue
            for key in results.get('img_fields', ['img']):
                H, W, _ = results[key].shape 
                results[key] = cv2.resize(results[key], (W * 2, H * 2))
                results[key] = match_histograms(image=results[key], reference=ref, multichannel=self.multichannel)
                results[key] = cv2.resize(results[key], (W, H)).astype(np.uint8)
            return results
        else:
            return results