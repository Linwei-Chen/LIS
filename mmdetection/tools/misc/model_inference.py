from glob import glob
from mmdet.apis import init_detector, inference_detector
import cv2
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
('bicycle', 'chair', 'dining table', 'bottle', 'motorcycle', 'car', 'tv', 'bus')
('bicycle', 'car', 'motorcycle', 'bus', 'bottle', 'chair', 'dining table', 'tv')

color_dict = {"bicycle": [67, 159, 36],
 "chair": [204, 43, 41], 
 "diningtable": [145, 104, 190],
 "dining table": [145, 104, 190], 
 "bottle": [135, 86, 75], 
 "motorbike": [245, 128, 6], 
 "motorcycle": [245, 128, 6],  
 "car": [53, 119, 181], 
 "tvmonitor": [219, 120, 195],  
 "tv": [219, 120, 195],  
 "bus": [127, 127, 127]
#  "bus": [204, 43, 41], 
 }
 
def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0, 
                       color=(72, 101, 241)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    # def show_result(self,
    #             img,
    #             result,
    #             score_thr=0.3,
    #             bbox_color=(72, 101, 241),
    #             text_color=(72, 101, 241),
    #             mask_color=None,
    #             thickness=2,
    #             font_size=13,
    #             win_name='',
    #             show=False,
    #             wait_time=0,
    #             out_file=None):
    img = model.show_result(
                img,
                result,
                score_thr=score_thr,
                show=False,
                wait_time=wait_time,
                win_name=title,
                bbox_color=color,
                mask_color=color,
                text_color=(255, 255, 255),
                thickness=2,
                font_size=15, 
                color_dict=color_dict
                )
    return img



# exp_name = 'RGBDark'
# exp_name = 'RAWDark_inRGB'
# exp_name = 'CameraDark'
exp_name = 'HE'
# exp_name = 'ZD'
# exp_name = 'EG'
# exp_name = 'RN_SGN30'
# exp_name = 'GN_SGN30'
# exp_name = 'ZD_SGN30'
# exp_name = 'EG_SGN30'
# exp_name = 'HE_SGN30'
# exp_name = 'RN'
# exp_name = 'GN'


# img_list_path = '/home/ubuntu/2TB/dataset/LOD/Camera_Dark/Enhance_result/EnlightenGAN/JPEGImages/*.JPG'
# img_list_path = '/home/ubuntu/2TB/dataset/LOD/Camera_Dark/Enhance_result/Zero-DCE/JPEGImages/*.JPG'
# img_list_path = '/home/ubuntu/2TB/dataset/LOD/Camera_Dark/Enhance_result/GladNet/JPEGImages/*.JPG'
# img_list_path = '/home/ubuntu/2TB/dataset/LOD/Camera_Dark/Enhance_result/Retinex-net/JPEGImages/*.JPG'
img_list_path = f'/home/ubuntu/2TB/dataset/LOD/Camera_Dark/{exp_name}/JPEGImages/*.JPG'
# img_list_path = f'/home/ubuntu/2TB/dataset/LOD/RAW_Dark/NoRatio/JPEGImages/*.png'
# img_list_path = f'/home/ubuntu/2TB/dataset/LOD/RGB_Dark/Merge/JPEGImages/*.png'
# img_list_path = f'/home/ubuntu/2TB/dataset/LOD/Camera_Dark/HE/JPEGImages/*.JPG'
# img_list_path = f'/home/ubuntu/2TB/dataset/LOD/Camera_Dark/HE/JPEGImages/*.JPG'
# img_list_path = f'/home/ubuntu/2TB/dataset/LOD/Camera_Dark/Merge/JPEGImages/*.JPG'
img_list = glob(img_list_path)
# img_list = [i  for i in img_list if '1908' in i]
# img_list = [i  for i in img_list if '2516' in i]
# img_list = [i  for i in img_list if '3654' in i]
print(len(img_list))

# rgb_dark_list = glob(f'/home/ubuntu/2TB/dataset/LOD/RGB_Dark/Merge/JPEGImages/*.png')

config_file = '/home/ubuntu/code/LowLight/mmdetexp/mask_rcnn_r50_fpn_caffe_lod.py'
# config_file = '/home/ubuntu/code/LowLight/mmdetexp/mask_rcnn_r50adad_fpn_caffe_lod.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN600x400/Formal/r50_fpn_nothing_baseline_8cls40.8_12E/latest.pth'
# checkpoint_file = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN600x400/Formal/r50_fpn_bs8_8cls_UN_NI10-100_Noise_Inv_.01_C1-4_Cleanloss_NoDetach_AdaConv3333T_8cls40.8_12E/latest.pth'
# checkpoint_file = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN600x400/Formal/r50_fpn_bs8_8cls_UN_NI10-300_Noise_Inv_.01_C1-4_Cleanloss_NoDetach_AdaConv3333T_8cls40.8_12E/latest.pth'
# checkpoint_file = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN600x400/Formal/r50_fpn_bs8_8cls_UN_NI10-300_AdaConv3333T_PTFS=4_8cls40.8_12E/latest.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image

"""
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/vibe-env/bin/python /home/ubuntu/code/mmdetection/tools/misc/model_inference.py
"""

check_point_dir = osp.split(checkpoint_file)[0]
exp_dir = osp.join(check_point_dir, exp_name)
print(exp_dir)
if not osp.exists(exp_dir):
    os.mkdir(exp_dir)
for img_path in tqdm(img_list):
    # img = cv2.imread(img_path)
    # img = np.clip(img + np.random.randn(800, 1200, 3) * 0.0, 0, 255)
    # res = inference_detector(model, img)
    res = inference_detector(model, img_path)
    # print(len(res))
    # for i in res:
        # print(i)
        # exit()
    img_name = osp.split(img_path)[1]
    paired_img_path = '/home/ubuntu/2TB/dataset/LOD/RGB_Dark/Merge/JPEGImages/' + img_name
    # _img = cv2.imread(paired_img_path)
    _img = cv2.imread(img_path)
    # _img = np.clip(_img + np.random.randn(*_img.shape) * 40, 0, 255)
    
    show_img = show_result_pyplot(model, _img, res, score_thr=0.5)
    # show_img = show_result_pyplot(model, img_path, res, score_thr=0.5)
    img_name = osp.split(img_path)[1]
    img_save_path = osp.join(exp_dir, img_name)
    # print(show_img, img_save_path)
    cv2.imwrite(img_save_path, show_img)
    # cv2.imwrite(img_save_path, _img)
