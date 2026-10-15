# IJCV2023 Instance Segmentation in the Dark

![img](https://github.com/Linwei-Chen/LIS/blob/main/static/segindark-pic-center.gif)

The implementation of our IJCV 2023 paper "[Instance Segmentation in the Dark](https://arxiv.org/abs/2304.14298)".

Interested readers are also referred to an insightful [Note](https://zhuanlan.zhihu.com/p/656570195) about this work in Zhihu.

## News! 📰

- **2025.10: Full open-source release.**
  - 🤗 Dataset (four image types + annotations, CC BY 4.0): [Linwei-Chen/LIS](https://huggingface.co/datasets/Linwei-Chen/LIS)
  - 🤗 Model weights (verified checkpoints + configs): [Linwei-Chen/LIS-models](https://huggingface.co/Linwei-Chen/LIS-models)
  - 🔌 Plug-and-play single file [`lis_pluggable.py`](./lis_pluggable.py): `import` it and all custom modules register into stock `mmdet==2.15.1` — no source replacement needed.
- **CVPR 2024 challenge: [Low-light Object Detection and Instance Segmentation Challenge](https://codalab.lisn.upsaclay.fr/competitions/17833)** — 4th International Workshop on Physics-Based Vision Meets Deep Learning (PBDL), CVPR 2024, Seattle.

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

## Highlight✨

- We propose an **adaptive weighted downsampling (AWD)** layer, **smooth-oriented convolutional block (SCB)** and **disturbance suppression learning (DSL)** to address the high-frequency disturbance within deep features that occurred in very low light. Interestingly, they also benefit the normal-lit instance segmentation.
- We exploit the potentials of **RAW-input design** for low-light instance segmentation and leverage a low-light RAW synthetic pipeline to generate realistic low-light RAW images from existing datasets, which facilitates end-to-end training.
- We collect a real-world low-light dataset with precise pixel-wise instance-level annotations, namely **LIS**, which covers more than two thousand scenes and can serve as a benchmark for instance segmentation in the dark. On LIS, our approach outperforms state-of-the-art competitors in terms of both segmentation accuracy and inference speed by a large margin.

## Method Overview 

<img src="https://github.com/Linwei-Chen/LIS/blob/main/static/overview.png" width="512px">

The adaptive weighted downsampling (AWD) layer, smooth-oriented convolutional block (SCB), and disturbance suppression loss are designed to reduce the feature disturbance caused by noise, and the low-light RAW synthetic pipeline is employed to facilitate end-to-end training of instance segmentation on RAW images.

## Dataset Overview

![img](https://github.com/Linwei-Chen/LIS/blob/main/static/dataset.png)

Four image types (long-exposure normal-light and short-exposure low-light images in both RAW and sRGB formats) are captured for each scene.

## Installation

This repo is based on [MMDetection](https://github.com/open-mmlab/mmdetection) **v2.15.1**.

```bash
conda create -n lis python=3.8 -y
conda activate lis
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install pycocotools scipy matplotlib opencv-python rawpy exifread
# Usage A runs against the bundled mmdetection/ fork directly;
# for Usage B additionally: pip install mmdet==2.15.1
```

## Usage

### Usage A — full fork (recommended; exactly the paper setup)

The bundled [`mmdetection/`](./mmdetection) is our modified mmdet 2.15.1 — the authoritative implementation behind every number in the paper (incl. PointRend / Mask2Former variants). Run from the repo root:

```bash
cd mmdetection
# evaluate a released checkpoint against the paper numbers
python tools/test.py ../Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2LIS.py \
    <ckpt.pth> --eval bbox segm
# train
python tools/train.py ../Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2LIS.py
```

Method locations: AWD → [`backbones/CustomConv.py`](./mmdetection/mmdet/models/backbones/CustomConv.py) · SCB → [`backbones/resnet.py` (SmoothPriorBottleneck)](./mmdetection/mmdet/models/backbones/resnet.py) and [`backbones/convnext.py`](./mmdetection/mmdet/models/backbones/convnext.py) · DSL → [`detectors/mask_rcnn.py` (MaskRCNNNoiseInv)](./mmdetection/mmdet/models/detectors/mask_rcnn.py) · RAW low-light noise synthesis → [`datasets/pipelines/noisemodel/dark_noising.py`](./mmdetection/mmdet/datasets/pipelines/noisemodel/dark_noising.py).

The synthetic pipeline from RGB to RAW in low-light conditions can be used in any pipeline as follows:

```python
dict(type='AddNoisyImg', model='PGRU', camera='CanonEOS5D4',
     cfa='bayer', use_255=True, pre_adjust_brightness=False,
     mode='unprocess_addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
```

### Usage B — plug-and-play single file

If you prefer a stock `pip install mmdet==2.15.1`, just `import lis_pluggable` before building the model; every custom module (backbones / FPN / detector / pipeline transforms) registers automatically:

```python
import lis_pluggable  # one line: registers AWD / SCB / DSL + noise pipeline into mmdet registries

from mmdet.apis import init_detector, inference_detector  # then use mmdet as usual
```

`lis_pluggable.py` covers the core method line (R50 / ConvNeXt backbones, `MaskRCNNNoiseInv`, noise synthesis; camera noise parameters in [`camera_params/`](./camera_params)). PointRend / Mask2Former variants require the full fork (Usage A).

### Reparameterization (zero-overhead inference)

The SCB low-pass prior branch can be folded into the adjacent convolution at inference time:

```bash
python tools/reparameter.py IN_CKPT OUT_CKPT
```

### Dataset tools

[`dataset_tools/`](./dataset_tools) contains the camera ISP and RAW reading utilities used to build the dataset (`ISP.py`, `read_RAW.py`, `read_RAW2RGB.py`).

## Data Preparation

Download from 🤗 [Linwei-Chen/LIS](https://huggingface.co/datasets/Linwei-Chen/LIS) (CC BY 4.0), or the original [Baidu Drive](https://pan.baidu.com/s/1iusYmp8iVMWsmCsJo7sWaw) (code: IJCV) / [Google Drive](https://drive.google.com/drive/folders/1KpC82G_H1CI35lmnB2LYr9aK3FQcahAC?usp=share_link). Place data as:

```
data/
├── LIS/                              # ann file_name is './JPEGImages/N.png' relative to each group dir
│   ├── RGB-normal/JPEGImages/        # odd ids, 1200×800 png (long-exposure sRGB)
│   ├── RGB-dark/JPEGImages/          # even ids, png (short-exposure, rendered)
│   ├── RAW-normal/JPEGImages/        # odd ids, png (RAW rendered to sRGB)
│   ├── RAW-dark/JPEGImages/          # even ids, png (RAW rendered, no-ratio)
│   └── annotations/
│       ├── lis_coco_png_train.json    # normal-light ids (1561)
│       ├── lis_coco_png_test+1.json   # dark side ('+1' = dark ids), test (669)
│       ├── lis_coco_png_traintest+1.json
│       └── ...                        # png/JPG series, coco.json variants, sub-ratio subsets
└── coco/                              # COCO 2017 train/val (only for training configs)
```

- All configs use repo-relative data roots (`data/LIS/`, `data/coco/`) — run scripts from the repo root, or adjust `data_root`.
- **Privacy note**: a small number of scenes (179 pairs — close-up parked cars with potentially readable license plates, plus a few screens kept out of caution) were **excluded** from the released images and annotations; the exact id list ships with the HF dataset card.
- Revised annotations (LIS 1.1, see [issue #23](https://github.com/Linwei-Chen/LIS/issues/23#issue-2971196455)): [Baidu Drive](https://pan.baidu.com/s/1ZDXaeDYHPpcuJ0PR3yviKQ?pwd=IJCV) (code: IJCV).
- Original camera sensor RAW (CR2) will be released at 🤗 [Linwei-Chen/LIS-raw](https://huggingface.co/Linwei-Chen/LIS-raw) after privacy screening.

## Model Zoo

Weights are hosted at 🤗 [Linwei-Chen/LIS-models](https://huggingface.co/Linwei-Chen/LIS-models). Results on LIS test set (dark side, `lis_coco_png_test+1.json`):

| Model              | Backbone   | Train set | Seg AP   | Box AP   | Config | CKPT |
| ------------------ | ---------- | --------- | -------- | -------- | ------ | ---- |
| Mask R-CNN         | R50        | COCO      | 19.8     | 22.8     | [config](./Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_COCO80.py) | [HF](https://huggingface.co/Linwei-Chen/LIS-models/blob/main/mask_rcnn_r50_fpn_COCO8%3D19.8.pth) |
| Mask R-CNN (Ours)  | R50        | Syn COCO  | **31.8** | **37.6** | [config](./Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2LIS.py) | [HF](https://huggingface.co/Linwei-Chen/LIS-models/blob/main/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2LIS%3D31.8.pth) |
| Mask R-CNN         | ConvNeXt-T | COCO      | 23.7     | 27.9     | [config](./Configs/mask_rcnn_convnext-t_p4_w7_fpn_COCO2LIS.py) | [HF](https://huggingface.co/Linwei-Chen/LIS-models/blob/main/mask_rcnn_convnext-t_p4_w7_fpn_COCO2LIS%3D23.7.pth) |
| Mask R-CNN (Ours)  | ConvNeXt-T | Syn COCO  | **36.8** | **42.7** | [config](./Configs/mask_rcnn_convnext-t_p4_w7_fpn_AWD_SCB_DSL_SynCOCO2LIS.py) | [HF](https://huggingface.co/Linwei-Chen/LIS-models/blob/main/mask_rcnn_convnext-t_p4_w7_fpn_AWD_SCB_DSL_SynCOCO2LIS%3D36.8.pth) |
| PointRend          | R50        | COCO      | 20.6     | 23.5     | [config](./Configs/pointrend_r50_fpn_caffe_COCO2LIS.py) | [HF](https://huggingface.co/Linwei-Chen/LIS-models/blob/main/pointrend_r50_fpn_caffe_COCO2LIS%3D20.6.pth) |
| PointRend (Ours)   | R50        | Syn COCO  | **32.8** | **37.1** | [config](./Configs/pointrend_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2LIS.py) | [HF](https://huggingface.co/Linwei-Chen/LIS-models/blob/main/pointrend_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2LIS%3D32.8.pth) |
| Mask2Former        | R50        | COCO      | 21.4     | 22.9     | [config](./Configs/mask2former_r50_COCO2LIS.py) | [HF](https://huggingface.co/Linwei-Chen/LIS-models/blob/main/mask2former_r50_SynCOCO2LIS%3D21.4.pth) |
| Mask2Former (Ours) | R50        | Syn COCO  | **35.6** | **37.8** | [config](./Configs/mask2former_r50_caffe_AWD_SCB_DSL_SynCOCO2LIS.py) | [HF](https://huggingface.co/Linwei-Chen/LIS-models/blob/main/mask2former_r50_AWD_SCB_DSL_SynCOCO2LIS%3D37.8.pth) |

We do not tune hyperparameters like loss weights. Further adjusting the hyperparameters should lead to improvement.

On normal-light COCO val set (AWD/SCB/DSL also help in daylight):

| Model                  | Backbone | Train | Seg AP | Box AP |
| ---------------------- | -------- | ----- | ------ | ------ |
| Mask R-CNN             | R50      | COCO  | 34.4   | 38.0   |
| Mask R-CNN+AWD+SCB+DSL | R50      | COCO  | **36.1** | **39.5** |

For future research, we suggest using COCO as train set and the **whole LIS** as test set (full results: Table 5 of the paper).

## Citation

If you use our dataset or code for research, please cite this paper and our [previous work](https://github.com/ying-fu/LODDataset/tree/main):

```bibtex
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

```bibtex
@inproceedings{Hong2021Crafting,
	title={Crafting Object Detection in Very Low Light},
	author={Yang Hong, Kaixuan Wei, Linwei Chen, Ying Fu},
	booktitle={BMVC},
	year={2021}
}
```

## License

This project is released under the [LICENSE](./LICENSE). The LIS dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgement

We thank [MMDetection](https://github.com/open-mmlab/mmdetection) for the open-source toolbox, and [LOD](https://github.com/ying-fu/LODDataset/tree/main) for inspiring the dataset design.

## Contact

If you find any problem, please feel free to contact me (Linwei at chenlinwei@bit.edu.cn). A brief self-introduction (including your name, affiliation, and position) is required, if you would like to get in-depth help from me. I'd be glad to talk with you if more information (e.g. your personal website link) is attached.
