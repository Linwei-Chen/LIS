from genericpath import exists
from math import fabs, pi
import os
from time import time
from unicodedata import name

from cv2 import decolor
from torch.nn.modules import loss
from .aux_modules import *
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
import copy
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from PIL import Image
import matplotlib.pyplot as plt
from .aux_modules import *
from .multiscale_discriminator import *
from .lsid import LSID
import time

colors = plt.cm.jet(np.linspace(0.0, 1.00, 256))
# for i in range(len(colors)):
    # temp = colors[i][0]
    # colors[i][0] = colors[i][2]
    # colors[i][2] = temp

@DETECTORS.register_module()
class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)


@DETECTORS.register_module()
class MaskRCNNNoiseInv(MaskRCNN):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self, **args):
        super().__init__(**args)
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      return_proposal=False,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        backbone_x, x = self.extract_feat(img, return_backbone=True)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        if return_proposal:
            return losses, backbone_x, x, proposal_list
        else:
            return losses, backbone_x, x

    def copy_self(self, load_from):
        # return None
        # print('copy')
        raise Exception
        self.cp = copy.deepcopy(self)
        teacher_ckpt = load_from
        print('load teacher param from:', teacher_ckpt)
        load_checkpoint(self.cp, teacher_ckpt, map_location='cuda')

    # @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            losses = dict()
            noisy_img = kwargs.pop('noisy_img')
            clean_losses, clean_backbone_x, clean_x = self.forward_train(img, img_metas, **kwargs)
            noisy_losses, noisy_backbone_x, noisy_x = self.forward_train(noisy_img, img_metas, **kwargs)
            losses['noise_inv_loss'] = 0
            for idx, (_noisy, _clean) in enumerate(zip(noisy_backbone_x[:], clean_backbone_x[:])):
                losses['noise_inv_loss'] += 0.01 * torch.clamp(F.mse_loss(input=_noisy, target=_clean, reduction='none'), max=0.1, min=0).mean()
            for k in clean_losses: losses[f'clean_{k}'] = clean_losses[k]
            for k in noisy_losses: losses[f'{k}'] = noisy_losses[k]
            for k in losses: 
                if isinstance(losses[k], int): 
                    losses.pop(k)
            return losses
        else:
            # return self.forward_test(img, img_metas, **kwargs)
            return super().forward_test(img, img_metas, **kwargs)

    def extract_feat(self, img, return_backbone=True):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            if return_backbone:
                backbone_x, fpn_x = self.neck(x, return_backbone)
                return backbone_x, fpn_x
            else:
                fpn_x = self.neck(x, return_backbone)
                return fpn_x
        else:
            return x
        

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        # img_residual = F.interpolate(self.unet(img), (img.size(-2), img.size(-1)))
        # img = img + img_residual
        x = self.extract_feat(img, return_backbone=False)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        def feature2image(feat, save_dir, name, normalize=False):
            _, _, h, w = feat.shape
            feat = feat.mean(dim=1).repeat(3, 1, 1)
            feat = F.interpolate(feat[None, ], size=(400, 600), mode='bilinear')[0]
            save_img = feat.permute(1, 2, 0).cpu().numpy()
            if normalize:
                save_img = (save_img - save_img.min()) / (save_img.max() - save_img.min()) * 255
            else:
                save_img = save_img * 255
                save_img = np.clip(save_img, 0, 255)
            save_img = save_img.astype(np.uint8)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            colored = np.zeros(shape=(save_img.shape[0], save_img.shape[1], 3)).astype(np.uint8)
            for i in range(3):
                for num in np.unique(save_img):
                    # print(num)
                    # print(colored[save_img == int(num)])
                    colored[:,:,i][save_img[:, :, i] == int(num)] = colors[int(num)][i] * 255.
            # Image.fromarray(save_img).save(f'{save_dir}/{name}_.jpg')
            Image.fromarray(colored).save(f'{save_dir}/{name}.png')
            pass

        def tensor2img(t, save_dir, name):
            norm = torch.tensor([103.530, 116.280, 123.675])[None, :, None, None].to(t.device)
            t += norm
            img = t[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            Image.fromarray(img).save(f'{save_dir}/{name}.png')

        # img = self.unet(img) + img
        backbone_x, x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
