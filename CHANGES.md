# Changelog

## 2025.10 — Repository restructure & full open-source release

### Repo-level changes

- **Added [`lis_pluggable.py`](./lis_pluggable.py)**: single-file plug-and-play module for stock `mmdet==2.15.1`. Registers AWD (AdaDConv), SCB (SmoothPriorBottleneck / SmoothPriorConvNeXtBlock), the DSL detector (`MaskRCNNNoiseInv`), the enhanced FPN (registers over stock `FPN`, adding optional `return_backbone`), and the RAW low-light noise pipeline (`NoiseModel` / `AddNoisyImg` / `hist_match`). Camera noise parameters moved to `camera_params/`.
  - Excluded on purpose: `FasterRCNNNoiseInv` (dead code referencing a non-existent `CosLoss`, crashes on import), `PointRendNoiseInv` / `Mask2Former` variants (reproduce via Usage A fork).
- **Configs/**: replaced all machine-specific absolute paths (`/data3/...`, `/home/ubuntu/...`) with repo-relative `data/LIS/` and `data/coco/`; replaced local pretrained-checkpoint paths with official OpenMMLab download URLs; fixed `LIS2LIS` config having a duplicated `ann_file` key (the effective challenge-annotation entry also mismatched the declared class order — the safe `lis_coco_png_test+1.json` entry is now effective, the challenge line kept as a comment).
- **`mmdetection/mmdet/datasets/pipelines/noisemodel/dark_noising.py`**: camera parameter directory no longer hard-codes the author's machine; defaults to `camera_params/` next to the module.
- **`utils/` split**: RAW/ISP utilities → `dataset_tools/`; `reparameter.py` → `tools/reparameter.py` (now takes IN/OUT checkpoint arguments instead of hard-coded paths).
- **Removed** `mmdetection_custom_part/` (redundant byte-identical copy of the fork's `models/` subtree), `update.sh`, stray `__pycache__` / `.DS_Store` / `temp_test.py`.
- **README rewritten**: installation, both usages, data layout, HF model zoo, privacy note.

### mmdetection fork w.r.t. stock v2.15.1 (unchanged logic, documented for transparency)

- `models/backbones/resnet.py`: adds `SmoothPriorBottleneck` (SCB), `ResNetAdaD`, `ResNetAdaDSmoothPrior` (AWD); `ResNet.__init__` builds frozen stages with stock `Bottleneck` and unfrozen stages with the configured custom block.
- `models/backbones/convnext.py`: backports ConvNeXt from a later mmdet (absent in 2.15.1) and adds `ConvNeXtAdaD` / `ConvNeXtAdaDSmoothPrior`.
- `models/backbones/CustomConv.py`: adds `AdaDConv` (adaptive weighted downsampling), `LLPFConv`, `GaussianBlurConv`, `LaplacianConv` prior filters.
- `models/necks/fpn.py`: `FPN.forward` gains an optional `return_backbone` output (needed by DSL feature-consistency loss); identical behavior when unset.
- `models/detectors/mask_rcnn.py` / `point_rend.py` / `faster_rcnn_noise_inv.py`: adds `MaskRCNNNoiseInv` / `PointRendNoiseInv` / `FasterRCNNNoiseInv` — clean/noisy dual forward with clamped MSE feature consistency (DSL), plus `aux_modules.py`, `multiscale_discriminator.py`, `lsid.py` helpers.
- `datasets/pipelines/noisemodel/dark_noising.py`: adds `LoadDarkPair` for paired RAW-dark / RAW-normal loading, alongside the noise-model transforms.
