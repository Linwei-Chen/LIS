import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_head
from .single_stage import SingleStageDetector
from .aux_modules import *
import copy

@DETECTORS.register_module()
class YOLACT(SingleStageDetector):
    """Implementation of `YOLACT <https://arxiv.org/abs/1904.02689>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 segm_head,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLACT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)
        self.segm_head = build_head(segm_head)
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
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

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # convert Bitmap mask or Polygon Mask to Tensor here
        gt_masks = [
            gt_mask.to_tensor(dtype=torch.uint8, device=img.device)
            for gt_mask in gt_masks
        ]

        x = self.extract_feat(img)

        cls_score, bbox_pred, coeff_pred = self.bbox_head(x)
        bbox_head_loss_inputs = (cls_score, bbox_pred) + (gt_bboxes, gt_labels,
                                                          img_metas)
        losses, sampling_results = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        segm_head_outs = self.segm_head(x[0])
        loss_segm = self.segm_head.loss(segm_head_outs, gt_masks, gt_labels)
        losses.update(loss_segm)

        mask_pred = self.mask_head(x[0], coeff_pred, gt_bboxes, img_metas,
                                   sampling_results)
        loss_mask = self.mask_head.loss(mask_pred, gt_masks, gt_bboxes,
                                        img_metas, sampling_results)
        losses.update(loss_mask)

        # check NaN and Inf
        for loss_name in losses.keys():
            assert torch.isfinite(torch.stack(losses[loss_name]))\
                .all().item(), '{} becomes infinite or NaN!'\
                .format(loss_name)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        feat = self.extract_feat(img)
        det_bboxes, det_labels, det_coeffs = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]

        segm_results = self.mask_head.simple_test(
            feat,
            det_bboxes,
            det_labels,
            det_coeffs,
            img_metas,
            rescale=rescale)

        return list(zip(bbox_results, segm_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError(
            'YOLACT does not support test-time augmentation')
            


    def __init__(self, **kargs):
        super().__init__(**kargs)
        # self.ISP = ISPNet()
        # self.eval_teacher = eval_teacher
        # # Build teacher model
        # if isinstance(teacher_config, str):
        #     teacher_config = mmcv.Config.fromfile(teacher_config)
        # self.teacher_model = build_detector(teacher_config['model'])
        # if teacher_ckpt is not None:
        #     load_checkpoint(
        #         self.teacher_model, teacher_ckpt, map_location='cpu')

    # def copy_self(self):
        # self.cp = copy.deepcopy(self)

    
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
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

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # convert Bitmap mask or Polygon Mask to Tensor here
        gt_masks = [
            gt_mask.to_tensor(dtype=torch.uint8, device=img.device)
            for gt_mask in gt_masks
        ]

        backbone_x, x = self.extract_feat(img)

        cls_score, bbox_pred, coeff_pred = self.bbox_head(x)
        bbox_head_loss_inputs = (cls_score, bbox_pred) + (gt_bboxes, gt_labels,
                                                          img_metas)
        losses, sampling_results = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        segm_head_outs = self.segm_head(x[0])
        loss_segm = self.segm_head.loss(segm_head_outs, gt_masks, gt_labels)
        losses.update(loss_segm)

        mask_pred = self.mask_head(x[0], coeff_pred, gt_bboxes, img_metas,
                                   sampling_results)
        loss_mask = self.mask_head.loss(mask_pred, gt_masks, gt_bboxes,
                                        img_metas, sampling_results)
        losses.update(loss_mask)

        # check NaN and Inf
        for loss_name in losses.keys():
            assert torch.isfinite(torch.stack(losses[loss_name]))\
                .all().item(), '{} becomes infinite or NaN!'\
                .format(loss_name)

        # return losses, x, bbox_pred, mask_pred
        return losses, backbone_x, x

    def extract_feat(self, img, return_backbone=True):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            backbone_x, fpn_x = self.neck(x, return_backbone)
        return backbone_x, fpn_x

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
            ori_img = kwargs.pop('ori_img')

            clean_losses, clean_backbone_x, clean_x = self.forward_train(img, img_metas, **kwargs)
            noisy_losses, noisy_backbone_x, noisy_x = self.forward_train(noisy_img, img_metas, **kwargs)

            # noise invariance loss
            noise_inv_loss = 0
            cos_loss = 0
            # losses['spatial_loss'] = 0
            # losses['channel_loss'] = 0
            # losses['sc_loss'] = 0
            losses['noise_inv_loss'] = 0
            c_t, s_t, c_s_t = 1.0, 1.0, 1.0
            gamma = 0.01
            beta = 0.01
            # beta = 0.00001
            # for idx, (_noisy, _clean) in enumerate(zip(noisy_x, clean_x)):
            for idx, (_noisy, _clean) in enumerate(zip(noisy_backbone_x[:], clean_backbone_x[:])):
                _b, _c, _h, _w = _noisy.shape
                # n_c = _noisy.abs().mean(dim=(-1, -2), keepdim=True)
                # c_c = _clean.abs().mean(dim=(-1, -2), keepdim=True)
                # n_c_att = (_noisy.abs().mean(dim=(-1, -2), keepdim=True) / c_t).softmax(dim=1)
                # c_c_att = (_clean.abs().mean(dim=(-1, -2), keepdim=True) / c_t).softmax(dim=1)

                # n_s = _noisy.abs().mean(dim=(1), keepdim=True)
                # c_s = _clean.abs().mean(dim=(1), keepdim=True)
                # n_s_att = (_noisy.abs().mean(dim=(1), keepdim=True) / s_t).reshape(_b, -1).softmax(dim=-1).reshape(_b, -1, _h, _w).detach()
                # c_s_att = (_clean.abs().mean(dim=(1), keepdim=True) / s_t).reshape(_b, -1).softmax(dim=-1).reshape(_b, -1, _h, _w).detach()
                # print(n_s_att.shape, c_s_att.shape)

                # losses['spatial_loss'] += gamma * torch.clamp(F.mse_loss(n_c, c_c, reduction='mean'), max=1.0)
                # losses['channel_loss'] += gamma * torch.clamp(F.mse_loss(n_s, c_s, reduction='mean'), max=1.0)
                
                # losses['sc_loss'] += gamma * (F.mse_loss(_noisy, _clean, reduction='none') * (n_s_att + c_s_att) * 0.5).mean(dim=(1, 2, 3))
                # losses['sc_loss'] += beta * torch.clamp((F.mse_loss(_noisy, _clean, reduction='none') * c_s_att).sum(dim=(1, 2, 3)) / _c , max=1.0) #* _h * _w 
                    # (F.mse_loss(_noisy, _clean.detach(), reduction='none') * (n_s_att + c_s_att) * 0.5).sum(dim=(1, 2, 3)) * _h * _w / _c, max=1.0)

                losses['noise_inv_loss'] += 0.01 * torch.clamp(F.mse_loss(input=_noisy, target=_clean, reduction='mean'), max=1., min=0)
                # losses['noise_inv_loss'] += .1 * torch.clamp(F.mse_loss(input=_noisy, target=_clean.detach(), reduction='mean'), max=1., min=0)
                # losses['noise_inv_loss'] += .1 * torch.clamp(F.mse_loss(input=self.AL[idx](_noisy), target=_clean, reduction='mean'), max=1., min=0)
                # losses['noise_inv_loss'] += beta * torch.clamp(F.mse_loss(input=_noisy, target=_clean, reduction='mean'), max=1.0) * _h * _w
            

            for k in clean_losses: losses[f'clean_{k}'] = clean_losses[k]
            for k in noisy_losses: losses[f'{k}'] = noisy_losses[k]
            return losses
            # return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)


    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        backbone_f, feat = self.extract_feat(img)
        det_bboxes, det_labels, det_coeffs = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]

        segm_results = self.mask_head.simple_test(
            feat,
            det_bboxes,
            det_labels,
            det_coeffs,
            img_metas,
            rescale=rescale)

        return list(zip(bbox_results, segm_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError(
            'YOLACT does not support test-time augmentation')