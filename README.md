# IJCV2023 Instance Segmentation in the Dark

The implementation  of our IJCV2023 paper "Instance Segmentation in the Dark".

Interested readers are also referred to an insightful [Note](https://zhuanlan.zhihu.com/) about this work in Zhihu. (TODO) 



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



## ✨Highlight

-  We propose an adaptive weighted downsampling layer, smooth-oriented convolutional block and disturbance suppression learning to address the high-frequency disturbance within deep features that occurred in very low light. Interestingly, they also benefit the normal-lit instance segmentation.

- We exploit the potentials of RAW-input design for low-light instance segmentation and leverage a low-light RAW synthetic pipeline to generate realistic low-light RAW images from existing datasets, which facilitates end-to-end training.
- We collect a real-world low-light dataset with precise pixel-wise instance-level annotations, namely LIS, which covers more than two thousand scenes and can serve as a benchmark for instance segmentation in the dark. On LIS, our approach outperforms state-of-the-art competitors in terms of both segmentation accuracy and inference speed by a large margin.



## Method Overview

<img src="https://github.com/Linwei-Chen/LIS/blob/main/static/overview.png" width="512px">

The adaptive weighted downsampling (AWD) layer, smooth-oriented convolutional block (SCB), and disturbance suppression loss are designed to reduce the feature disturbance caused by noise, and the low-light RAW synthetic pipeline is employed to facilitate end-to-end training of instance segmentation on RAW images.



## Dataset Overview

![img](https://github.com/Linwei-Chen/LIS/blob/main/static/dataset.png)

Four image types (long-exposure normal-light and short-exposure low-light images in both RAW and sRGB formats) are captured for each scene.}



## Code Usage

### Installation

Our code is based on MMDetection.

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation) for installation and [dataset_prepare.md](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.



### Pretrained Model

Results are reported on COCO val set.

| Model                  | Backbone | Train | Seg AP | Box AP | Config                                                       | CKPT                                                         |
| ---------------------- | -------- | ----- | ------ | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Mask R-CNN             | R50      | COCO  | 34.4   | 38.0   | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth) |
| Mask R-CNN+AWD+SCB+DSL | R50      | COCO  | 36.1   | 39.5   | config                                                       | model                                                        |

Results are reported on LIS test set.

| Model      | Backbone | Train     | Seg AP | Box AP | Config | CKPT  |
| ---------- | -------- | --------- | ------ | ------ | ------ | ----- |
| Mask R-CNN | R50      | COCO      |        |        | config | model |
| Mask R-CNN | R50      | Syn COCO  |        |        | config | model |
| Mask R-CNN | R50      | LIS train |        |        | config | model |

For future research, we suggest using COCO as train set and the whole LIS as test set.



## Dataset Download





## Citation

If you use our dataset or code for research, please cite our papers:

```
@article{LIS,
	title={Instance Segmentation in the Dark},
	author={Chen, Linwei and Fu, Ying and Wei, Kaixuan and Zheng, Dezhi and Heide, Felix},
	journal={International Journal of Computer Vision},
	year={2023}
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
