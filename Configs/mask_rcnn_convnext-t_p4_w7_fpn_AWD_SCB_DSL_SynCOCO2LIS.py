ROOT = '~/mmdetection/configs/'
_base_ = [
    ROOT + '/_base_/models/mask_rcnn_r50_fpn.py',
    ROOT + '/_base_/datasets/coco_instance.py',
    ROOT + '/_base_/schedules/schedule_1x.py', 
    ROOT + '/_base_/default_runtime.py'
]


# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data3/chenlinwei/dataset/coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Resize', img_scale=(600, 400), keep_ratio=True, interpolate_mode='nearest'),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AddNoisyImg', model='PGRU', camera='CanonEOS5D4',
    # dict(type='NoiseModel', model='PGRU', camera='CanonEOS5D4',
    #     #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess'),
         cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess_addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
    #     #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'noisy_img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(600, 400),
        flip=False,
        transforms=[
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
    ann_file='/data3/chenlinwei/dataset/LOD/lis_coco_png_test+1.json',
    # ann_file='/data3/chenlinwei/dataset/LOD/lis_coco_png_traintest+1.json',
    # ann_file='/data3/chenlinwei/dataset/LOD/lis_coco_JPG_test+1.json',
    # ann_file='/data3/chenlinwei/dataset/LOD/lis_coco_JPG_traintest+1.json',
    img_prefix='/data3/chenlinwei/dataset/LOD/RAW_Dark/',
    # img_prefix='/data3/chenlinwei/dataset/LOD/RGB_Dark/',
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
# optimizer = dict(type='SGD', lr=0.02 * BATCHSIZE * GPU / 16, momentum=0.9, weight_decay=0.0001,
#     paramwise_cfg = dict(
#         custom_keys={
#             # 'llpf': dict(lr_mult=2.), 
#             # 'learnable_conv1': dict(lr_mult=2.),
#             # 'AdaD': dict(lr_mult=2.),
#             }))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[9, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)


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

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/convnext/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco/mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco_20220426_154953-050731f4.pth'
# please install mmcls>=0.22.0
# import mmcls.models to trigger register_module in mmcls
# custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

model = dict(
    # type='MaskRCNN',
    type='MaskRCNNNoiseInv',
    backbone=dict(
        _delete_=True,
        # type='ConvNeXt',
        # type='ConvNeXtAdaD',
        type='ConvNeXtAdaDSmoothPrior',
        arch='tiny',
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(
            num_classes=8,),
        mask_head=dict(
            num_classes=8,)),
)


optimizer = dict(
    _delete_=True,
    # constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6,
        'custom_keys':{
            'llpf': dict(lr_mult=10.), 
            # 'learnable_conv1': dict(lr_mult=2.),
            # 'AdaD': dict(lr_mult=2.),
            }
    })

# lr_config = dict(warmup_iters=1000, step=[27, 33])
# runner = dict(max_epochs=36)
lr_config = dict(warmup_iters=1000, step=[9, 11])
runner = dict(max_epochs=12)
# you need to set mode='dynamic' if you are using pytorch<=1.5.0
# fp16 = dict(loss_scale=dict(init_scale=512))