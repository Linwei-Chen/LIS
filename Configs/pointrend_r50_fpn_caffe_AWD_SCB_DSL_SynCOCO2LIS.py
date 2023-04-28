_base_ = './mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2LIS.py'

load_from = '/data3/chenlinwei/.cache/torch/point_rend_r50_caffe_fpn_mstrain_3x_coco_41.0_38.0.pth'

log_level = 'INFO'
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=False),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(600, 400), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='NoiseModel', model='PGRU', camera='CanonEOS5D4',
    dict(type='AddNoisyImg', model='PGRU', camera='CanonEOS5D4',
        #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess'),
         cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess_addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
        #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'ori_img', 'noisy_img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(600, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, interpolate_mode='nearest'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

coco = dict(
    classes=('bicycle', 'chair', 'dining table',
             'bottle', 'motorcycle', 'car', 'tv', 'bus'),
    type='CocoDataset',
    ann_file='/data3/chenlinwei/dataset/coco/annotations/instances_train2017.json',
    img_prefix='/data3/chenlinwei/dataset/coco/train2017/',
    pipeline=train_pipeline)

coco_val = dict(
    classes=('bicycle', 'chair', 'dining table',
             'bottle', 'motorcycle', 'car', 'tv', 'bus'),
    type='CocoDataset',
    ann_file='/data3/chenlinwei/dataset/coco/annotations/instances_val2017.json',
    img_prefix='/data3/chenlinwei/dataset/coco/val2017/',
    pipeline=test_pipeline)


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
    
BATCHSIZE = 8
GPU=1
optimizer = dict(type='SGD', lr=0.02 * BATCHSIZE * GPU / 16,
                 momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=BATCHSIZE,
    workers_per_gpu=BATCHSIZE,
    train=[coco, ] * 1,
    val=test_lod_coco,
    test=test_lod_coco)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

model = dict(
    type='PointRendNoiseInv',
    backbone=dict(
        # type='ResNet',
        # type='ResNetAdaD',
        type='ResNetAdaDSmoothPrior',
        ),
    roi_head=dict(
        type='PointRendRoIHead',
        mask_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='concat',
            roi_layer=dict(
                _delete_=True, type='SimpleRoIAlign', output_size=14),
            out_channels=256,
            featmap_strides=[4]),
        mask_head=dict(
            _delete_=True,
            type='CoarseMaskHead',
            num_fcs=2,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        point_head=dict(
            type='MaskPointHead',
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=8,
            coarse_pred_each_layer=True,
            loss_point=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            mask_size=7,
            num_points=14 * 14,
            oversample_ratio=3,
            importance_sample_ratio=0.75)),
    test_cfg=dict(
        rcnn=dict(
            subdivision_steps=5,
            subdivision_num_points=28 * 28,
            scale_factor=2)))