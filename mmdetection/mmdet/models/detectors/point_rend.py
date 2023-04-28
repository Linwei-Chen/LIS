from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
import os
from PIL import Image
import torch.nn.functional as F
import numpy as np
from .lsid import LSID

@DETECTORS.register_module()
class PointRend(TwoStageDetector):
    """PointRend: Image Segmentation as Rendering

    This detector is the implementation of
    `PointRend <https://arxiv.org/abs/1912.08193>`_.

    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(PointRend, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)


@DETECTORS.register_module()
class PointRendNoiseInv(PointRend):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self, **args):
        super().__init__(**args)
        # self.ISP = EmbedISPNet(in_channels=512, upsample_times=3)
        # self.ISP = EmbedISPNet(in_channels=256, upsample_times=2)
        # self.recover_decoder = Decoder(in_channels=256, upsample_times=2)
        self.noise_inv_list = [[], [], [], []]
        # self.MD = MultiscaleDiscriminator(2048, num_D=5, use_psp=True)

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
            # _dark = torch.tensor(np.random.uniform(0.01, 1.0)).to(img.device)
            # dark_noisy_img = noisy_img * _dark
            noisy_img = kwargs.pop('noisy_img')
            ori_img = kwargs.pop('ori_img')

            """
            norm = torch.tensor([103.530, 116.280, 123.675])[None, :, None, None].to(img.device)
            norm.requires_grad = False
            noisy_img += norm
            ori_img += norm
            img += norm

            # print('up', img[0].mean(dim=(-1, -2)), 'ori', ori_img[0].mean(dim=(-1, -2)), 'noise', noisy_img[0].mean(dim=(-1, -2)))
            # print(img[0].min(), ori_img[0].min(), noisy_img[0].min())
            noisy_img= self.ISP(noisy_img)
            # noisy_img = self.ISP(noisy_img)
            
            losses['ISP_loss'] = F.l1_loss(input=noisy_img / 255., target=ori_img / 255.)
            # noisy_img -= norm
            # ori_img -= norm
            # img -= norm
            return losses
            """
            # with torch.no_grad():
                # self.cp.eval()
                # clean_backbone_x, clean_x = self.cp.extract_feat(ori_img, return_backbone=True)
                # clean_backbone_x, clean_x = self.extract_feat(ori_img, return_backbone=True)
                # clean_backbone_x, clean_x = self.extract_feat(img, return_backbone=True)
                # proposal_list = self.cp.rpn_head.simple_test_rpn(clean_x, img_metas)
                # rpn_cls_score, rpn_bbox_pred = self.cp.rpn_head(clean_x)
                # def kd_loss(cls_score, bbox_pred, cls_score_t, bbox_pred_t):
                #     cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cp.rpn_head.cls_out_channels).sigmoid()
                #     cls_score_t = cls_score_t.permute(0, 2, 3, 1).reshape(-1, self.cp.rpn_head.cls_out_channels).sigmoid()
                #     # regression loss
                #     bbox_targets = bbox_targets.reshape(-1, 4)
                #     bbox_weights = bbox_weights.reshape(-1, 4)
                #     bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                # for i in rpn_cls_score:
                # for i in rpn_bbox_pred :
                    # print(i.shape, i)
                # print(proposal_list)
                # self.roi_head.simple_test(clean_x, proposal_list, img_metas, rescale=False)
                # print(clean_backbone_x.shape, clean_x.shape)
    
            clean_losses, clean_backbone_x, clean_x = self.forward_train(img, img_metas, **kwargs)
            noisy_losses, noisy_backbone_x, noisy_x = self.forward_train(noisy_img, img_metas, **kwargs)
            # noisy_losses, noisy_backbone_x, noisy_x = self.forward_train(img, img_metas, **kwargs)
            # noisy_losses, noisy_backbone_x, noisy_x = self.forward_train(dark_noisy_img, img_metas, **kwargs)

            # std = None
            # std = torch.tensor([103.530, 116.280, 123.675])[None, :, None, None].to(img.device)
            # std.requires_grad = False
            # noisy_img += std
            # ori_img += std
            # img += std

            # gain_x, denoised_x, normal_x, d_normal_x = self.ISP(feat=noisy_backbone_x[0], x=noisy_img, std=std)
            # gain_x, denoised_x, normal_x, d_normal_x = self.ISP(feat=noisy_backbone_x[0], x=dark_noisy_img, std=std)
            # losses['gain_x_loss'] = F.l1_loss(input=gain_x / 255., target=noisy_img / 255.)
            # losses['denoised_x_loss'] = F.l1_loss(input=denoised_x / 255., target=img / 255.)
            # losses['normal_x _loss'] = torch.clamp(F.l1_loss(input=normal_x / 255., target= ori_img / 255.), 0, +0.4)
            # losses['d_normal_x _loss'] = F.l1_loss(input=d_normal_x / 255., target=ori_img / 255.)
            
            # direct_normal_x = self.img_decoder(feat=noisy_backbone_x[3])
            # losses['d_normal_x _loss'] = F.mse_loss(input=(noisy_img + direct_normal_x) / 255., target=img / 255.)
            # losses['d_normal_x _loss'] = F.l1_loss(input=(noisy_img + direct_normal_x) / 255., target=img / 255.)
            # losses['d_normal_x _loss'] = 0.1 * F.mse_loss(input=direct_normal_x / 255., target=ori_img / 255.)
            # losses['d_denoised_x _loss'] = 100 * F.l1_loss(input=(noisy_img + direct_normal_x) / 255., target=img / 255.)
            # losses['denoise _loss'] = 1 * F.l1_loss(input=(direct_normal_x) / 255., target=img / 255.)

            

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

                # losses['noise_inv_loss'] += 0.01 * torch.clamp(F.mse_loss(input=_noisy, target=_clean, reduction='mean'), max=1., min=0)
                losses['noise_inv_loss'] += 0.01 * torch.clamp(F.mse_loss(input=_noisy, target=_clean, reduction='none'), max=0.1, min=0).mean()
                # losses['noise_inv_loss'] += .1 * torch.clamp(F.mse_loss(input=_noisy, target=_clean.detach(), reduction='mean'), max=1., min=0)
                # losses['noise_inv_loss'] += .1 * torch.clamp(F.mse_loss(input=self.AL[idx](_noisy), target=_clean, reduction='mean'), max=1., min=0)
                # losses['noise_inv_loss'] += beta * torch.clamp(F.mse_loss(input=_noisy, target=_clean, reduction='mean'), max=1.0) * _h * _w
            
            # losses['noise_inv_loss'] = 0
            # for _noisy, _clean in zip(noisy_backbone_x[:], clean_backbone_x[:]):
            #     # noise_inv_loss += .01 * F.mse_loss(input=_noisy, target=_clean, reduction='mean')
            #     losses['noise_inv_loss'] += .001 * torch.clamp(
            #         F.mse_loss(input=_noisy, target=_clean, reduction='mean'),
            #         # F.mse_loss(input=F.avg_pool2d(_noisy, kernel_size=2, stride=2), 
            #                 #    target=F.avg_pool2d(_clean, kernel_size=2, stride=2), reduction='mean'),
            #     #     # F.mse_loss(input=_noisy, target=_clean, reduction='none'),
            #     #     # F.l1_loss(input=_noisy, target=_clean, reduction='mean'),
            #     #     # F.mse_loss(input=_noisy, target=_clean.detach(), reduction='none'),
            #     #     # F.smooth_l1_loss(input=_noisy, target=_clean, reduction='mean'),
            #     #     # F.l1_loss(input=_noisy, target=_clean.detach(), reduction='mean'),
            #         max=1.0, min=0).mean()
                # cos_loss -= .01 * self.cos_loss(_noisy, _clean).mean()
            # losses['noise_inv_loss'] = noise_inv_loss
            # losses['cos_loss'] = cos_loss

            # losses['FA_loss'] = 0 
            # for _noisy, _clean in zip(noisy_backbone_x[:], clean_backbone_x[:]):
                # losses['FA_loss'] += 0.1 * self.fa_loss(_noisy, _clean)
            # GAN losses
            # gan_losses = self.MD.FPN_D_loss(noisy_backbone_x[3:], clean_backbone_x[3:])
            # for k in gan_losses: losses[f'{k}'] = 0.1 * torch.clamp(gan_losses[k], min=-1., max=1.)
            # for idx, (_noisy, _clean) in enumerate(zip(noisy_backbone_x[:], clean_backbone_x[:])):
            #     gan_losses = self.MD[idx].FPN_D_loss(fakes=[_noisy], reals=[_clean])
            #     for k in gan_losses: 
            #         if k not in losses: losses[f'{k}'] = 0
            #         if k == 'd_loss':
            #             losses[f'{k}'] += 0.1 * torch.clamp(gan_losses[k], min=-1., max=1.)
            #         else:
            #             losses[f'{k}'] += 0.01 * torch.clamp(gan_losses[k], min=-1., max=1.)

            # alpha = 1.5
            # for k in clean_losses: 
            #     if isinstance(clean_losses[k], list):
            #         clean_losses[k] = [alpha * i for i in clean_losses[k]]
            #     else:
            #         clean_losses[k] = alpha * clean_losses[k]
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
            backbone_x, fpn_x = self.neck(x, return_backbone)
        return backbone_x, fpn_x

    # def forward_dummy(self, img):
    #     # noisy_img = kwargs.pop('noisy_img')
    #     # ori_img = kwargs.pop('ori_img')
    #     with torch.no_grad():
    #         self.cp.eval()
    #         clean_backbone_x, clean_x = self.cp.extract_feat(img)
    #         print(clean_backbone_x[0].shape, clean_x[0].shape)

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        # print('before', img.mean(dim=(-1, -2)))
        # norm = torch.tensor([103.530, 116.280, 123.675])[None, :, None, None].to(img.device)
        # img = self.ISP(img + norm) - norm
        # print(img + norm))
        # print(img.mean(dim=(-1, -2)))
        # print((img + norm).mean(dim=(-1, -2)))

        def feature2image(feat, save_dir, name, normalize=False):
            _, _, h, w = feat.shape
            feat = feat.mean(dim=1).repeat(3, 1, 1)
            # feat = F.interpolate(feat[None, ], scale_factor=1, mode='bilinear', align_corners=True)[0]
            # feat = F.interpolate(feat[None, ], size=(400, 600), mode='nearest')[0]
            feat = F.interpolate(feat[None, ], size=(400, 600), mode='bilinear')[0]
            # print(vis_feature.shape)
            # std = torch.tensor([103.530, 116.280, 123.675])[None, :, None, None].to(img.device)
            # print(img_metas)
            # name = np.random.randint(low=0, high=100000)
            # name = os.path.splitext(os.path.split(img_metas[0]['ori_filename'])[1])[0]
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


        backbone_x, x = self.extract_feat(img)
        cal_diff = False
        if cal_diff:
            noisy_img = kwargs.pop('noisy_img')[0]
            # ori_img = kwargs.pop('ori_img')[0]
            # print(noisy_img.shape, img.shape)
            noisy_backbone_x, noisy_x = self.extract_feat(noisy_img)
            noise_inv_loss = 0
            for idx, (_noisy, _clean) in enumerate(zip(noisy_backbone_x[:], backbone_x[:])):
                # print(_noisy.mean())
                noise_inv_loss += F.mse_loss(input=_noisy, target=_clean.detach(), reduction='mean')
                # noise_inv_loss += F.l1_loss(input=_noisy, target=_clean.detach(), reduction='mean')
                self.noise_inv_list[idx].append(noise_inv_loss)
                print(f'{idx}:', sum(self.noise_inv_list[idx]) / len(self.noise_inv_list[idx]))
            img_name = os.path.splitext(os.path.split(img_metas[0]['ori_filename'])[1])[0]
            # save_dir = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN/diff_29.0'
            # save_dir = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN/Vis_UP_NI_RGB_Normal'
            save_dir = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN/Vis_UP_NI_AdaConv3333T_NoiseInv_RGB_Normal'
            # save_dir = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN/Noisy_Img'
            if False:
                for i in range(4):
                    feature2image((backbone_x[i] - noisy_backbone_x[i]).abs() * 1., save_dir, f'{img_name}_C{i}_L1diff', normalize=False)
                    feature2image(backbone_x[i], save_dir, f'{img_name}_C{i}_clean', normalize=True)
                    feature2image(noisy_backbone_x[i].abs(), save_dir, f'{img_name}_C{i}_noisy', normalize=True)
                # feature2image(noisy_backbone_x[3], save_dir, f'{img_name}_noisy', normalize=True)
                # feature2image(backbone_x[3], save_dir, f'{img_name}', normalize=True)
                tensor2img(noisy_img, save_dir, f'{img_name}_noisy_img')
                tensor2img(img, save_dir, f'{img_name}_img')

       
        vis = False
        if vis:
             # vis_feature = backbone_x[1].max(dim=1)[0].repeat(3, 1, 1)
            vis_feature = backbone_x[1]
            _, _, h, w = vis_feature.shape
            vis_feature = vis_feature[0, 0:1].reshape(-1).softmax(dim=0).reshape(-1, h, w).repeat(3, 1, 1)
            # vis_feature = backbone_x[0].softmax(dim=1).mean(dim=1).repeat(3, 1, 1)
            vis_feature = F.interpolate(vis_feature[None, ], scale_factor=4, mode='bilinear', align_corners=True)[0]
            # print(vis_feature.shape)
            # std = torch.tensor([103.530, 116.280, 123.675])[None, :, None, None].to(img.device)
            # print(img_metas)
            # name = np.random.randint(low=0, high=100000)
            name = os.path.splitext(os.path.split(img_metas[0]['ori_filename'])[1])[0]
            save_img = vis_feature.permute(1, 2, 0).cpu().numpy()
            save_img = save_img / (save_img.max() - save_img.min()) * 255
            save_img = save_img.astype(np.uint8)
            # save_dir = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN/27.0_bk2'
            save_dir = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN/31.2_bk0_0'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            colored = np.zeros(shape=(save_img.shape[0], save_img.shape[1], 3)).astype(np.uint8)
            # img[img < 75] = 0
            for i in range(3):
                # print(save_img.shape, colored.shape)
                for num in np.unique(save_img):
                    # print(num)
                    # print(colored[save_img == int(num)])
                    colored[:,:,i][save_img[:, :, i] == int(num)] = colors[int(num)][i] * 255.
            # Image.fromarray(save_img).save(f'{save_dir}/{name}_.jpg')
            Image.fromarray(colored).save(f'{save_dir}/{name}.png')
            # Image.fromarray(img[0] + ).save(f'{save_dir}/{name}')
            # print('\n', vis_feature.min(), vis_feature.max())
            # img[0] = self.ISP(img[0])
            # print(img[0].min(), img[0].max())
            # print(len(img))
            # print(img[0].shape)
            # print(img[0][0, ].permute(1, 2, 0).shape)
            # save_img = img[0][0, ].permute(
            #     1, 2, 0).cpu().numpy() * 50 + 110
            # save_img = save_img.astype(np.uint8)
            # Image.fromarray(save_img).save(
            #     f'/home/ubuntu/code/LowLight/mmdetexp/fasterRCNN/temp/{name}.jpg')

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)




@DETECTORS.register_module()
class PointRendUNet(PointRend):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self, **args):
        super().__init__(**args)
        self.unet = LSID(inchannel=3, block_size=1)
        

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
            # _dark = torch.tensor(np.random.uniform(0.01, 1.0)).to(img.device)
            # dark_noisy_img = noisy_img * _dark
            noisy_img = kwargs.pop('noisy_img')
            ori_img = kwargs.pop('ori_img')    

            noisy_img = self.unet(noisy_img) + noisy_img
            # losses['denoise_loss'] = F.l1_loss(input=noisy_img / 255., target=img / 255.)
            losses['denoise_mse_loss'] = F.mse_loss(input=noisy_img / 255., target=img / 255.)

            # clean_losses, clean_backbone_x, clean_x = self.forward_train(img, img_metas, **kwargs)
            noisy_losses, noisy_backbone_x, noisy_x = self.forward_train(noisy_img, img_metas, **kwargs)
            

            # for k in clean_losses: losses[f'clean_{k}'] = clean_losses[k]
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
            backbone_x, fpn_x = self.neck(x, return_backbone)
        return backbone_x, fpn_x

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'

        img = self.unet(img) + img
        backbone_x, x = self.extract_feat(img)
        cal_diff = False
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
