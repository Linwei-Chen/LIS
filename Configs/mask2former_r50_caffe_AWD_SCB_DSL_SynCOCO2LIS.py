ROOT = '/data3/chenlinwei/code/AdaptiveDet/mmdetection/configs/'
_base_ = [
    ROOT + '/_base_/datasets/coco_instance.py',
    ROOT + '/_base_/schedules/schedule_1x.py', 
    ROOT + '/_base_/default_runtime.py'
]


# dataset settings
dataset_type = 'CocoDataset'
# data_root = '/data3/chenlinwei/dataset/coco/'
data_root = '/data3/chenlinwei/dataset/coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

image_size = (608, 608)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        # poly2mask=False
        ),
    # dict(
    #     type='Resize',
    #     img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
    #                (1333, 768), (1333, 800)],
    #     multiscale_mode='value',
    #     keep_ratio=True),
    dict(type='Resize', img_scale=image_size, keep_ratio=True, interpolate_mode='nearest'),
    dict(type='RandomFlip', flip_ratio=0.5),
        dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        # recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), 
        # by_mask=True
        ),
    dict(type='Pad', size=image_size, 
        # pad_val=pad_cfg
        ),
    # dict(type='AddNoisyImg', model='PGRU', camera='CanonEOS5D4',
    dict(type='NoiseModel', model='PGRU', camera='CanonEOS5D4',
    #     #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess'),
         cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='unprocess_addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
    #     #  cfa='bayer', use_255=True, pre_adjust_brightness=False, mode='addnoise', dark_ratio=(1.0, 1.0), noise_ratio=(10, 100)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', 
            size_divisor=32, 
            # size=(608, 416)
            # size=(640, 448)
    ), ###
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'noisy_img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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

BATCHSIZE = 4
GPU = 2
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
load_from  = 'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_8x2_50e_coco/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

num_things_classes = 8
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='Mask2Former',
    backbone=dict(
        # type='ResNet',
        type='ResNetAdaDSmoothPrior',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        # with_cp=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=None)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(
    _delete_= True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
            'llpf': dict(lr_mult=10.), 
            # 'learnable_conv1': dict(lr_mult=2.),
            # 'AdaD': dict(lr_mult=2.),
        },
        norm_decay_mult=0.0))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(warmup_iters=10, step=[9, 11])
runner = dict(max_epochs=12)