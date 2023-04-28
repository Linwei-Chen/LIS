from skimage.exposure import match_histograms
import cv2
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