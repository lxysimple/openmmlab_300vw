# _base_ = ['../../../_base_/default_runtime.py']
_base_ = ["E:\\mmpose\\mmpose\\configs\\_base_\\default_runtime.py"]


# runtime
train_cfg = dict(max_epochs=60, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=2e-3,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=60,
        milestones=[40, 55],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='NME', rule='less', interval=1))

# codec settings
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=1.5)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        # 这里做了一点标准化
        # 源码见：mmpose/mmpose/models/data_preprocessors/data_preprocessor.py
        type='PoseDataPreprocessor', 
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict( # 给backbone四个stage定义超参数
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2, 
                block='BASIC',
                num_blocks=(4, 4), # 不同分支堆叠块不同，输出通道不同，这里第2个分支c=36是融合了第1个分支18个通道的
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict( 
                # 4个分支，每个分支有4个模块
                num_modules=3, # 这个应该是重复3次stage4吧
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144), # 最后一个分支融合了前3个通道信息，而且不同分支特征图的shape不同
                multiscale_output=True),
            # 自定义上采样算法
            upsample=dict(mode='bilinear', align_corners=False))
            ,
        init_cfg=dict(
            # type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w18'),
            type='Pretrained', checkpoint='E:\\mmpose\\checkpoint\\hrnetv2_w18_wflw_256x256_dark-3f8e0c2c_20210125.pth'),
    ),
    neck=dict(
        # 就是将各分支的feature map上采样融合到branch1分支中
        # 源码：mmpose/mmpose/models/necks/fmap_proc_neck.py
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        # 用少量卷积层从一个比较低分辨率图中产生68个关键点
        # 源码：mmpose/mmpose/models/heads/heatmap_heads/heatmap_head.py
        type='HeatmapHead',
        in_channels=270, # 就是18+36+72+144
        # out_channels=68, # 每个通道对应68个关键点
        out_channels=68, # 每个通道对应68个关键点
        deconv_out_channels=None,
        # conv_out_channels=(270, ), # 从源码中可知，每个中间层输出c=270
        conv_out_channels=(270, ), # 从源码中可知，每个中间层输出c=270
        conv_kernel_sizes=(1, ), 
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec), # 生成热图
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'Face300WDataset'
data_mode = 'topdown' 
# data_root = 'data/300w/'
data_root = 'E:\\mmpose\\data\\300w\\'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'), # 这里要标准化bbox(x,y,w,h)
    dict(type='RandomFlip', direction='horizontal'), # 这里要知道图片宽、高
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=60,
        scale_factor=(0.75, 1.25)),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True, # 一直让dataset对象在内存中
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/face_landmarks_300w_train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        # ann_file='annotations/face_landmarks_300w_valid.json',
        ann_file='annotations\\face_landmarks_300w_valid.json',
        # data_prefix=dict(img='images/'),
        data_prefix=dict(img='images\\'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='NME',
    norm_mode='keypoint_distance',
)
test_evaluator = val_evaluator
