# import os.path as osp

# dataset settings
dataset_type = 'CocoDataset'
# data_root = '/data3/chenlinwei/dataset/coco/'
data_root = '/data3/chenlinwei/dataset/coco/'

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    # please ensure using nearest, otherwise the noise level will be reduced, which is not fair.
    dict(type='Resize', img_scale=(600, 400), keep_ratio=True, interpolate_mode='nearest'), 
    dict(type='RandomFlip', flip_ratio=0.5),
    # AddNoisyImg for DSL
    # dict(type='AddNoisyImg', model='PGRU', camera='CanonEOS5D4',
    # dict(type='NoiseModel', model='PGRU', camera='CanonEOS5D4',
    #     #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess'),
        #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess_addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'noisy_img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(600, 400),
        flip=False,
        transforms=[
            # please ensure using nearest, otherwise the noise level will be reduced, which is not fair.
            dict(type='Resize', keep_ratio=True, 
                interpolate_mode='nearest'
            ),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


test_lod_coco = dict(
    classes=('bicycle', 'car', 'motorbike', 'bus',
             'bottle', 'chair', 'diningtable', 'tvmonitor'),
    type='CocoDataset',
    # ann_file='/data3/chenlinwei/dataset/LOD/lis_coco_png_test+1.json',
    # ann_file='/data3/chenlinwei/dataset/LOD/lis_coco_png_traintest+1.json',
    ann_file='/data3/chenlinwei/dataset/LOD/lis_coco_JPG_test+1.json',
    # ann_file='/data3/chenlinwei/dataset/LOD/lis_coco_JPG_traintest+1.json',
    # img_prefix='/data3/chenlinwei/dataset/LOD/RAW_Dark/',
    img_prefix='/data3/chenlinwei/dataset/LOD/RGB_Dark/',
    pipeline=test_pipeline)

coco = dict(
    classes=('bicycle', 'chair', 'dining table', 'bottle', 'motorcycle', 'car', 'tv', 'bus'),
    type='CocoDataset',
    ann_file='/data3/chenlinwei/dataset/coco/annotations/instances_train2017.json',
    # seg_prefix='/data3/chenlinwei/dataset/coco/stuffthingmaps/train2017/',
    img_prefix='/data3/chenlinwei/dataset/coco/train2017/',
    pipeline=train_pipeline)

coco_val = dict(
    classes=('bicycle', 'chair', 'dining table', 'bottle', 'motorcycle', 'car', 'tv', 'bus'),
    type='CocoDataset',
    ann_file='/data3/chenlinwei/dataset/coco/annotations/instances_val2017.json',
    img_prefix='/data3/chenlinwei/dataset/coco/val2017/',
    pipeline=test_pipeline)

BATCHSIZE = 8
GPU = 1
data = dict(
    samples_per_gpu=BATCHSIZE,
    workers_per_gpu=BATCHSIZE,
    train=[coco, ],
    # val=coco_val,
    val=test_lod_coco,
    # test=coco_val
    test=test_lod_coco
)

evaluation = dict(metric=['bbox', 'segm'])
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
            'llpf': dict(lr_mult=2.), 
            'learnable_conv1': dict(lr_mult=2.),
            'AdaD': dict(lr_mult=2.),
            }))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)


# default runtime
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dir'
load_from = '/data3/chenlinwei/.cache/torch/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408_segm_mAP-0.37.pth'
# from mmdetection

resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True

# model settings
model = dict(
    type='MaskRCNN',
    # type='MaskRCNNNoiseInv',
    backbone=dict(
        type='ResNet',
        # type='ResNetAdaD',
        # type='ResNetAdaDSmoothPrior',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        # style='pytorch',
        # with_cp=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
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
