
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=False),
    dict(type='LoadDarkPair', img_dir='/home/ubuntu/2TB/dataset/LOD/RAW_Dark/NoRatio/', ext='png'), # set RAW Dark path
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(600, 400), keep_ratio=True, interpolate_mode='nearest'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='AddNoisyImg', model='PGRU', camera='CanonEOS5D4',
    # dict(type='NoiseModel', model='PGRU', camera='CanonEOS5D4',
        #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess'),
        #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess_addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
        #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='Collect', keys=['img', 'noisy_img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    # dict(type='Collect', keys=['img', 'noisy_img', 'ori_img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(600, 400),
        # flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='Resize', keep_ratio=True, interpolate_mode='nearest'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


lod_dark = dict(
    # coco order
    # classes = ('bicycle', 'car', 'motorbike', 'bus', 'bottle', 'chair', 'diningtable', 'tvmonitor'),
    # type='VOCCustomDataset',
    classes=('bicycle', 'chair', 'diningtable', 'bottle',
             'motorbike', 'car', 'tvmonitor', 'bus'),
    # type='LODDataset', # for png
    type='CocoDataset',
    # ann_file='/home/ubuntu/2TB/dataset/LOD/lis_coco_png_train+1.json',
    ann_file='/home/ubuntu/2TB/dataset/LOD/lis_coco_png_train.json',
    img_prefix='/home/ubuntu/2TB/dataset/LOD/RAW_Normal_Gamma/',
    # img_format='png',
    pipeline=train_pipeline)


test_lod_coco = dict(
    # or coco name in lod order
    # classes = ('bicycle', 'chair', 'dining table', 'bottle', 'motorcycle', 'car', 'tv', 'bus'),
    # classes=('bicycle', 'chair', 'diningtable', 'bottle', 'motorbike', 'car', 'tvmonitor', 'bus'),
    # train on coco eval on LOD
    # classes=('bicycle', 'car', 'motorbike', 'bus',
            #  'bottle', 'chair', 'diningtable', 'tvmonitor'),
    classes=('bicycle', 'chair', 'diningtable', 'bottle',
             'motorbike', 'car', 'tvmonitor', 'bus'),
    type='CocoDataset',
    ann_file='/home/ubuntu/2TB/dataset/LOD/lis_coco_png_test+1.json',
    ann_file='/home/ubuntu/2TB/dataset/LOD/lis_coco_png_raw_dark_testonly_challenge.json',
    img_prefix='/home/ubuntu/2TB/dataset/LOD/RAW_Dark/NoRatio/',
    pipeline=test_pipeline)

BATCHSIZE = 1
GPU = 1

data = dict(
    samples_per_gpu=BATCHSIZE,
    workers_per_gpu=10,
    # train=[noisy_voc, d2n_voc, ori_voc],
    # train=[ori_voc, ori_voc],
    # train=[lod_dark, test_lis_normal] * 1,
    train=[lod_dark,] * 1, 
    # train=[lod_raw_normal] * 3,
    # train=[lod_normal,] * 3, 
    # train=[lod, lod],
    val=test_lod_coco,
    test=test_lod_coco)

evaluation = dict(metric=['bbox', 'segm'])
# evaluation = dict(interval=1, metric='bbox')
# evaluation = dict(interval=1, metric='mAP')


# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# Important: The default learning rate in config files is
# for 8 GPUs and 2 img/gpu (batch size = 8*2 = 16).
# According to the linear scaling rule, you need to set the learning rate
# proportional to the batch size if you use different GPUs or images per GPU,
# e.g., lr=0.01 for 4 GPUs * 2 imgs/gpu and lr=0.08 for 16 GPUs * 4 imgs/gpu.


# optimizer
optimizer = dict(type='SGD', lr=0.02 * BATCHSIZE * GPU / 16, momentum=0.9, weight_decay=0.0001, 
    paramwise_cfg = dict(
            custom_keys={
                # 'AdaD': dict(lr_mult=2.)
            }))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[28, 34])
# runner = dict(type='EpochBasedRunner', max_epochs=36)

# default runtime
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/ubuntu/code/LowLight/mmdetexp/fasterRCNN/r50_fpn_2x_bs2_8cls_PGBRU'
load_from = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN600x400/PT_r50_fpn_bs4_8cls_UN_NI10-100_COCO_NoiseInv_LLPF_Prior_8cls40.8_12E/latest.pth' #coco Pretrain

resume_from = None
workflow = [('train', 1)]


# model settings
model = dict(
    type='MaskRCNNNoiseInv',
    # type='MaskRCNN',
    backbone=dict(
        # type='ResNetAdaD',
        type='ResNetAdaDSmoothPrior',
        with_cp=True,
        # type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        # style='pytorch',
        style='caffe',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),

        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe'
            )),
    neck=dict(
        # type='FPN',
        type='FreqFusionCARAFEFPN',
        use_high_pass=True,
        use_low_pass=True,
        lowpass_kernel=5,
        highpass_kernel=3,
        compress_ratio=8,
        feature_align=True,
        semi_conv=True,
        use_dyedgeconv=False,
        feature_align_group=4,
        hf_att=False,

        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
