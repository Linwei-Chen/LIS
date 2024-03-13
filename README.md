# IJCV2023 Instance Segmentation in the Dark

![img](https://github.com/Linwei-Chen/LIS/blob/main/static/segindark-pic-center.gif)

The implementation  of our IJCV 2023 paper "[Instance Segmentation in the Dark](https://arxiv.org/abs/2304.14298)".

Interested readers are also referred to an insightful [Note](https://zhuanlan.zhihu.com/p/656570195) about this work in Zhihu. (TODO) 

**Abstract**

Existing instance segmentation methods are primarily designed for high-visibility inputs, and their performance degrades drastically in extremely low-light environments.
In this work, we take a deep look at instance segmentation in the dark and introduce several techniques that substantially boost the low-light inference accuracy.
Our method design is motivated by the observation that noise in low-light images introduces high-frequency disturbances to the feature maps of neural networks, thereby significantly degrading performance.
To suppress this ``feature noise", we propose a novel learning method that relies on an adaptive weighted downsampling layer, a smooth-oriented convolutional block, and disturbance suppression learning.
They can reduce feature noise during downsampling and convolution operation, and enable the model to learn disturbance-invariant features, respectively.
Additionally, we find that RAW images with high bit-depth can preserve richer scene information in low-light conditions compared to typical camera sRGB outputs, thus supporting the use of RAW-input algorithms. Our analysis indicates that high bit-depth can be critical for low-light instance segmentation.
To tackle the lack of annotated RAW datasets, we leverage a low-light RAW synthetic pipeline to generate realistic low-light data. 
Furthermore, to support this line of work, we capture a real-world low-light instance segmentation dataset.
It contains more than two thousand paired low/normal-light images with instance-level pixel-wise annotations.
Without any image preprocessing, we achieve satisfactory performance on instance segmentation in very low light (4~\% AP higher than state-of-the-art competitors), meanwhile opening new opportunities for future research.



## News! 📰

### [CVPR 2024 challenge：Low-light Object Detection and Instance Segmentation Challenge](https://codalab.lisn.upsaclay.fr/competitions/17833)

- 4th International Workshop on Physics-Based Vision Meets Deep Learning (PBDL) in Conjunction with CVPR 2024, Seattle, WA, USA.
- The Low-light Object Detection and Instance Segmentation track starts now! We release validation data and training data. Check out [this page](https://codalab.lisn.upsaclay.fr/competitions/17833) and prepare the submission!
- More challenges from the CVPR 2024 workshop PBDL can be found at [this link](https://pbdl-ws.github.io/pbdl2024/challenge/index.html)!

### Important dates

- 2024.02.20 Challenge site online
- 2024.02.21 Release of train data (paired images) and validation data (inputs only)
- 2024.03.01 Validation server online
- 2024.04.23 Final test data release (inputs only)
- 2024.04.30 Test submission deadline
- 2024.05.05 Fact sheets and code/executable submission deadline
- 2024.05.10 Preliminary test and rating results release to participants



## Highlight✨

-  We propose an adaptive weighted downsampling layer, smooth-oriented convolutional block and disturbance suppression learning to address the high-frequency disturbance within deep features that occurred in very low light. Interestingly, they also benefit the normal-lit instance segmentation.

- We exploit the potentials of RAW-input design for low-light instance segmentation and leverage a low-light RAW synthetic pipeline to generate realistic low-light RAW images from existing datasets, which facilitates end-to-end training.
- We collect a real-world low-light dataset with precise pixel-wise instance-level annotations, namely LIS, which covers more than two thousand scenes and can serve as a benchmark for instance segmentation in the dark. On LIS, our approach outperforms state-of-the-art competitors in terms of both segmentation accuracy and inference speed by a large margin.



## Method Overview 

<img src="https://github.com/Linwei-Chen/LIS/blob/main/static/overview.png" width="512px">

The adaptive weighted downsampling (AWD) layer, smooth-oriented convolutional block (SCB), and disturbance suppression loss are designed to reduce the feature disturbance caused by noise, and the low-light RAW synthetic pipeline is employed to facilitate end-to-end training of instance segmentation on RAW images.



## Dataset Overview 

![img](https://github.com/Linwei-Chen/LIS/blob/main/static/dataset.png)

Four image types (long-exposure normal-light and short-exposure low-light images in both RAW and sRGB formats) are captured for each scene.



## Code Usage

### Installation

Our code is based on [MMDetection](https://github.com/open-mmlab/mmdetection).

Please refer to [get_started.md](https://github.com/Linwei-Chen/LIS/blob/main/mmdetection/docs/get_started.md) for installation and [dataset_prepare.md](https://github.com/Linwei-Chen/LIS/blob/main/mmdetection/docs/1_exist_data_model.md) for dataset preparation.



### Pretrained Model

Results are reported on LIS test set.

| Model              | Backbone   | Train set | Seg AP   | Box AP   | Config                                                       | CKPT                                                         |
| ------------------ | ---------- | --------- | -------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Mask R-CNN         | R50        | COCO      | 19.8     | 22.8     | [config](https://github.com/Linwei-Chen/LIS/tree/main/Configs) | [model](https://pan.baidu.com/s/16GdAxJ7GX3I9v8Mtm2xpIQ) (IJCV) |
| Mask R-CNN (Ours)  | R50        | Syn COCO  | **31.8** | **37.6** | [config](https://github.com/Linwei-Chen/LIS/tree/main/Configs) | [model](https://pan.baidu.com/s/16GdAxJ7GX3I9v8Mtm2xpIQ) (IJCV) |
| Mask R-CNN         | ConvNeXt-T | COCO      | 23.7     | 27.9     | [config](https://github.com/Linwei-Chen/LIS/tree/main/Configs) | [model](https://pan.baidu.com/s/1f9Lo8qtRjoLqXyKm-wy2RA) (IJCV) |
| Mask R-CNN (Ours)  | ConvNeXt-T | Syn COCO  | **36.8** | **42.7** | [config](https://github.com/Linwei-Chen/LIS/tree/main/Configs) | [model](https://pan.baidu.com/s/1Ugc3LNHxHmjAJmaLGpouNg) (IJCV) |
| PointRend          | R50        | COCO      | 20.6     | 23.5     | [config](https://github.com/Linwei-Chen/LIS/tree/main/Configs) | [model](https://pan.baidu.com/s/16GdAxJ7GX3I9v8Mtm2xpIQ) (IJCV) |
| PointRend (Ours)   | R50        | Syn COCO  | **32.8** | **37.1** | [config](https://github.com/Linwei-Chen/LIS/tree/main/Configs) | [model](https://pan.baidu.com/s/16GdAxJ7GX3I9v8Mtm2xpIQ) (IJCV) |
| Mask2Former        | R50        | COCO      | 21.4     | 22.9     | [config](https://github.com/Linwei-Chen/LIS/tree/main/Configs) | [model](https://pan.baidu.com/s/16GdAxJ7GX3I9v8Mtm2xpIQ) (IJCV) |
| Mask2Former (Ours) | R50        | Syn COCO  | **35.6** | **37.8** | [config](https://github.com/Linwei-Chen/LIS/tree/main/Configs) | [model](https://pan.baidu.com/s/16GdAxJ7GX3I9v8Mtm2xpIQ) (IJCV) |

We do not tune hyperparameters like loss weights. Further adjusting the hyperparameters should lead to improvement.

For future research, we suggest using COCO as train set and the **whole LIS** as test set.

| Model              | Backbone   | Train set | Seg AP   | Box AP   |
| ------------------ | ---------- | --------- | -------- | -------- |
| Mask R-CNN         | R50        | COCO      | 19.8     | 22.8     |
| Mask R-CNN (Ours)  | R50        | Syn COCO  | **27.2** | **33.3** |
| Mask R-CNN         | ConvNeXt-T | COCO      | 19.7     | 24.2     |
| Mask R-CNN (Ours)  | ConvNeXt-T | Syn COCO  | **32.6** | **39.1** |
| PointRend          | R50        | COCO      | 17.3     | 20.7     |
| PointRend (Ours)   | R50        | Syn COCO  | **27.3** | **32.0** |
| Mask2Former        | R50        | COCO      | 19.0     | 20.7     |
| Mask2Former (Ours) | R50        | Syn COCO  | **31.1** | **34.1** |



Results are reported on normal-light COCO val set.

| Model                  | Backbone | Train | Seg AP | Box AP | Config                                                       | CKPT                                                         |
| ---------------------- | -------- | ----- | ------ | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Mask R-CNN             | R50      | COCO  | 34.4   | 38.0   | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth) |
| Mask R-CNN+AWD+SCB+DSL | R50      | COCO  | 36.1   | 39.5   | [config](https://github.com/Linwei-Chen/LIS/tree/main/Configs) | [model](https://pan.baidu.com/s/16GdAxJ7GX3I9v8Mtm2xpIQ) (IJCV) |



## Dataset Download

Download images including RAW-dark, RAW-normal, RGB-dark, RGB-normal, and annotations from [Baidu Drive](https://pan.baidu.com/s/1XSV2CQZ7SWOvKvSgI7pH5Q) (extraction code: IJCV)  or [Google Drive](https://drive.google.com/drive/folders/1KpC82G_H1CI35lmnB2LYr9aK3FQcahAC?usp=share_link).

If the links are not available, please feel free to contact me.

LIS dataset should follow folder structure bellow:

```
├── LIS
│   ├── RGB-normal
│   │   ├── JPEGImages
│   │   │   ├──1.png
│   │   │   ├──3.png
│   │   │   ├──5.png
│   │   │   ├──...
│   ├── RGB-dark
│   │   ├── JPEGImages
│   │   │   ├──2.JPG
│   │   │   ├──4.JPG
│   │   │   ├──6.JPG
│   │   │   ├──...
│   ├── RAW-normal
│   │   ├── JPEGImages
│   │   │   ├──...
│   ├── RAW-dark
│   │   ├── JPEGImages
│   │   │   ├──...
│   ├── annotations
│   │   ├──lis_coco_JPG_train+1.json # w/ '+1' indicates labels for dark images
│   │   ├──lis_coco_JPG_test+1.json
│   │   ├──lis_coco_JPG_traintest+1.json  
│   │   ├──lis_coco_png_train.json # w/o '+1' indicates labels for normal images
│   │   ├──lis_coco_png_test.json
│   │   ├──lis_coco_png_traintest.json
│   │   ├──lis_coco_png_train+1.json
│   │   ├──lis_coco_png_test+1.json
│   │   ├──lis_coco_png_traintest+1.json
```

Original RAW files are avaliable in our [previous work](https://github.com/ying-fu/LODDataset/tree/main).



## Citation

If you use our dataset or code for research, please cite this paper and our [previous work](https://github.com/ying-fu/LODDataset/tree/main):

```
@article{2023lis,
  title={Instance Segmentation in the Dark},
  author={Chen, Linwei and Fu, Ying and Wei, Kaixuan and Zheng, Dezhi and Heide, Felix},
  journal={International Journal of Computer Vision},
  volume={131},
  number={8},
  pages={2198--2218},
  year={2023},
  publisher={Springer}
}
```

```
@inproceedings{Hong2021Crafting,
	title={Crafting Object Detection in Very Low Light},
	author={Yang Hong, Kaixuan Wei, Linwei Chen, Ying Fu},
	booktitle={BMVC},
	year={2021}
}
```



## Contact

If you find any problem, please feel free to contact me (Linwei at  chenlinwei@bit.edu.cn). A brief self-introduction (including your name, affiliation, and position) is required, if you would like to get in-depth help from me. I'd be glad to talk with you if more information (e.g. your personal website link) is attached.
