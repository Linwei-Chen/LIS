# Changelog

## 2025-10 — Repository restructure & full open-source release

### Repo-level changes

- **Added [`lis_pluggable.py`](./lis_pluggable.py)**: single-file plug-and-play module for stock `mmdet==2.15.1`. Registers AWD (AdaDConv), SCB (SmoothPriorBottleneck / SmoothPriorConvNeXtBlock), the DSL detector (`MaskRCNNNoiseInv`), the enhanced FPN (registers over stock `FPN`, adding optional `return_backbone`), and the RAW low-light noise pipeline (`NoiseModel` / `AddNoisyImg`). Camera noise parameters moved to `camera_params/`.
  - Excluded on purpose: `FasterRCNNNoiseInv` (dead code whose constructor references a non-existent `CosLoss` — it imports but crashes on instantiation), `PointRendNoiseInv` / `Mask2Former` variants (reproduce via Usage A fork).
- **Configs/**: replaced all machine-specific absolute paths (`/data3/...`, `/home/ubuntu/...`) with repo-relative `data/LIS/` and `data/coco/`; replaced local pretrained-checkpoint paths with official OpenMMLab download URLs; fixed `LIS2LIS` config having a duplicated `ann_file` key (the effective challenge-annotation entry also mismatched the declared class order — the safe `lis_coco_png_test+1.json` entry is now effective, the challenge line kept as a comment).
- **Configs (training reproducibility fixes)**: the three baseline training configs (`*_COCO2LIS.py` for R50 / ConvNeXt-T / PointRend) had `Collect` keys referencing `noisy_img` / `ori_img` that are never produced when the commented-out noise transforms are disabled — training would `KeyError` on the first sample; keys restored to `['img', 'gt_bboxes', 'gt_labels', 'gt_masks']`. The `LIS2LIS` training `img_dir` pointed to `RAW_Dark/NoRatio/` (author's working layout); the released `RAW_Dark/` is already the NoRatio variant, so the extra layer was dropped.
- **`mmdetection/mmdet/datasets/pipelines/noisemodel/dark_noising.py`**: camera parameter directory no longer hard-codes the author's machine; defaults to `camera_params/` next to the module.
- **`utils/` split**: RAW/ISP utilities → `dataset_tools/`; `reparameter.py` → `tools/reparameter.py` (now takes IN/OUT checkpoint arguments instead of hard-coded paths).
- **Removed** `mmdetection_custom_part/` (redundant byte-identical copy of the fork's `models/` subtree), `update.sh`, stray `__pycache__` / `.DS_Store` / `temp_test.py`.
- **README rewritten**: installation, both usages, data layout, HF model zoo, privacy note.

### mmdetection fork w.r.t. stock v2.15.1 (unchanged logic, documented for transparency)

- `mmdet/__init__.py`: mmcv compatibility range is `1.3.8 <= mmcv <= 1.4.8` (training originally used 1.4.0; 1.4.8 is the last release of the 1.4 line with prebuilt cu113/torch1.11 wheels, and is the recommended/re-evaluated version — API-stable within 1.4.x).

- `models/backbones/resnet.py`: adds `SmoothPriorBottleneck` (SCB), `ResNetAdaD`, `ResNetAdaDSmoothPrior` (AWD); `ResNet.__init__` builds frozen stages with stock `Bottleneck` and unfrozen stages with the configured custom block.
- `models/backbones/convnext.py`: backports ConvNeXt from a later mmdet (absent in 2.15.1) and adds `ConvNeXtAdaD` / `ConvNeXtAdaDSmoothPrior`.
- `models/backbones/CustomConv.py`: adds `AdaDConv` (adaptive weighted downsampling), `LLPFConv`, `GaussianBlurConv`, `LaplacianConv` prior filters.
- `models/necks/fpn.py`: `FPN.forward` gains an optional `return_backbone` output (needed by DSL feature-consistency loss); identical behavior when unset.
- `models/detectors/mask_rcnn.py` / `point_rend.py` / `faster_rcnn_noise_inv.py`: adds `MaskRCNNNoiseInv` / `PointRendNoiseInv` / `FasterRCNNNoiseInv` — clean/noisy dual forward with clamped MSE feature consistency (DSL), plus `aux_modules.py`, `multiscale_discriminator.py`, `lsid.py` helpers.
- `datasets/pipelines/noisemodel/dark_noising.py`: adds `LoadDarkPair` for paired RAW-dark / RAW-normal loading, alongside the noise-model transforms.

## Label ordering note (do not "fix" the classes order)

The `classes=` tuples in the LIS configs are ordered for the **label space the
models were trained in**, not the json `categories` id order. mmdet resolves
`cat_ids = coco.get_cat_ids(cat_names=self.CLASSES)` — a name->id lookup — so
training, evaluation and per-class AP printing are all consistent with each
other. Removing or re-sorting `classes` silently falls back to the 80-class
COCO `CLASSES` and breaks the pipeline (names like `diningtable` do not match
COCO's `dining table`). Verified by re-evaluation: checkpoint results match
the paper with the original tuples. The early-draft `coco.json`/`coco+1.json`
use a genuinely different id mapping and must not be mixed (see dataset card).

## Category-name misalignment in the json `categories` blocks (2025-10 addendum)

Instance-by-instance visual verification established the **true semantics** of
the 0-based category ids across all `lis_coco_*` files:
`0=bicycle, 1=car, 2=motorbike, 3=bus, 4=bottle, 5=chair, 6=diningtable, 7=tvmonitor`.
The `categories/name` blocks inside the released jsons claim a different order
(`0=bicycle, 1=chair, 2=diningtable, 3=bottle, 4=motorbike, 5=car, 6=tvmonitor,
7=bus`) and are therefore **misaligned with the true semantics** — e.g. id-1
instances are visually cars, not chairs. This also explains reports of
"mislabeled" visualization overlays: rendering with the json name block prints
wrong class names (the annotations themselves are correct).

Nothing needs fixing for reproduction: mmdet binds ids to names via a name->id
lookup, and the config `classes=` tuples already follow the true id order, so
training, evaluation and all published numbers are self-consistent. The json
name blocks are deliberately kept as released — renaming them would break
compatibility with the published checkpoints. See the HF dataset card for the
full explanation, per-class statistics, and example overlays rendered with
true class names.
