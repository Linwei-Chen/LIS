from math import e
from re import X

import numpy
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmcv.runner import BaseModule, auto_fp16
import torch
import torch.nn.functional as F
import torch.nn as nn
from .multiscale_discriminator import MultiscaleDiscriminator
from numpy import random
from .aux_modules import *
from PIL import Image
import copy
from ..backbones.CustomConv import BilateralFilter

@DETECTORS.register_module()
class FasterRCNNNoiseInv(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.cosloss = CosLoss()
        self.cp = None

    def copy_self(self):
        raise Exception
        self.cp = copy.deepcopy(self)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
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
        backbone_x, x = self.extract_feat(img)

        losses = dict()

        has_gt_mask = []
        _img_metas = []
        _gt_labels, _gt_bboxes = [], []
        for idx, label in enumerate(gt_labels):
            if len(label) < 1:
                continue
            else:
                has_gt_mask.append(idx)
                _gt_labels.append(gt_labels[idx])
                _gt_bboxes.append(gt_bboxes[idx])
                _img_metas.append(img_metas[idx])
        has_gt_mask = torch.LongTensor(has_gt_mask).to(img.device)
        gt_labels = _gt_labels
        gt_bboxes = _gt_bboxes
        img_metas = _img_metas

        if len(has_gt_mask) < 1:
            return losses, x

        filtered_x = list(x)
        if len(has_gt_mask) == x[0].size(0):
            pass
        else:
            for i in range(len(filtered_x)):
                filtered_x[i] = filtered_x[i].index_select(
                    dim=0, index=has_gt_mask).contiguous()
        # filtered_x = tuple(filtered_x)
        # print(filtered_x[0].shape)

        # RPN forward and loss
        if self.with_rpn:
            '''
            RPNHead(
                (loss_cls): CrossEntropyLoss()
                (loss_bbox): L1Loss()
                (rpn_conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (rpn_cls): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
                (rpn_reg): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
                )'''
            # print(self.rpn_head)
            # exit()
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                filtered_x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        roi_losses = self.roi_head.forward_train(filtered_x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses, backbone_x, x

    def forward_denoise_train(self,
                              img,
                              img_metas,
                              gt_bboxes,
                              gt_labels,
                              gt_bboxes_ignore=None,
                              gt_masks=None,
                              proposals=None,
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
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format. (abusolute coordinates)

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
        backbone_x, x = self.extract_feat(img, denoise=True)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
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
        # print(losses.keys())
        return losses, backbone_x, x

    def resize_img(self, img):
        # print(len(img), img[0].shape)
        for i in range(len(img)):
            new_size = (img[i].size(-2) // 2, img[i].size(-1) // 2)
            img[i] = F.interpolate(img[i], size=new_size,
                                   mode='bilinear', align_corners=True)
        return img

    def gen_att_map(self, img, gt_bboxes, use_centerness=False, **kwargs):
        att_maps = []
        for i, bbox in zip(img, gt_bboxes):
            h, w = i.size(-2), i.size(-1)
            bbox = bbox.round().int()

            if use_centerness:
                att_map = torch.zeros(1, 1, h, w).to(img.device)
                for b in bbox:
                    att_map[:, :, b[1]:b[3], b[0]:b[2]] = 1.0
            else:
                pass
            att_maps.append(att_map)
        return torch.cat(att_maps, dim=0)

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def extract_feat(self, img, return_backbone=True):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            backbone_x, fpn_x = self.neck(x, return_backbone)
        return backbone_x, fpn_x

    @auto_fp16(apply_to=('img', ))
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
            ori_img = kwargs.pop('ori_img')
            clean_losses, clean_backbone_x, clean_x = self.forward_train(img, img_metas, **kwargs)
            noisy_losses, noisy_backbone_x, noisy_x = self.forward_train(noisy_img, img_metas, **kwargs)
            noise_inv_loss = 0
            for _noisy, _clean in zip(noisy_backbone_x, clean_backbone_x):
                noise_inv_loss += .01 * torch.clamp(
                    F.smooth_l1_loss(
                        input=_noisy, target=_clean, reduction='none'),
                    max=0.1, min=0).mean()
            losses['noise_inv_loss'] = noise_inv_loss # / len(noisy_x)

            
            for k in clean_losses:
                losses[f'clean_{k}'] = clean_losses[k]
            for k in noisy_losses:
                losses[f'{k}'] = noisy_losses[k]
            return losses
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        backbone_x, x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
