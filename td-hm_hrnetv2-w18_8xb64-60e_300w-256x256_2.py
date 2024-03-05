# _base_ = ["E:\\mmpose\\mmpose\\configs\\_base_\\default_runtime.py"]
# data_root_300w='E:/mmpose/data/300w/'
# data_root_300vw='E:/mmpose/data/300vw/'

_base_ = ["/home/xyli/mmpose/configs/_base_/default_runtime.py"]

data_root_300w='/home/xyli/data/300w'
data_root_300vw='/home/xyli/data/300vw'
data_root_300vw_blur='/home/xyli/data'
data_root_300vw_deblur='/home/xyli'


# runtime
train_cfg = dict(max_epochs=80, val_interval=1)


# my optimizer
# optim_wrapper = dict(
#     optimizer=dict(
#         type='Adam',
#         lr=2e-5, # 2e-5最合适
#     ),
#     paramwise_cfg=dict(
#         custom_keys={
#             # 0最合适
#             'backbone.conv1': dict(lr_mult=0),
#             'backbone.bn1': dict(lr_mult=0),
#             'backbone.bn2': dict(lr_mult=0),
#             'backbone.layer1': dict(lr_mult=0),

#         }
#     )
# )

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',

    lr=2e-3, # 0.002
    # lr=2e-6, 
))

# learning policy
param_scheduler = [
    # lr=lr+b, at each iter in [0,500]; lr=2e-3 at iter=500
    dict(
        # type='LinearLR', begin=0, end=500, start_factor=0.001,

        # mmpose has 1 GPU, but I have 8 GUPs
        type='LinearLR', begin=0, end=60, start_factor=0.001,  
        by_epoch=False
    ),  # warm-up

    # lr=lr*0.1, at each epoch in [40,55]
    dict(
        type='MultiStepLR',
        begin=0,
        # end=60,
        end=80,
        # milestones=[40, 55],
        milestones=[50, 75],
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
# 这里应该是加载所有模块的权重
# resume = True # 是否基于上次训练状态开始
# load_from = '/home/xyli/checkpoint/hrnetv2_w18_300w_256x256-eea53406_20211019.pth' # ubuntu
# load_from = 'E:/mmpose/checkpoint/hrnetv2_w18_300w_256x256-eea53406_20211019.pth' # windows
load_from = '/home/xyli/checkpoint/hrnetv2_w18_300w_256x256-eea53406_20211019.pth' # ubuntu
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
            upsample=dict(mode='bilinear', align_corners=False)),

            # 预训练参数，只加载backbone权重用于迁移学习
            init_cfg=dict(
                # 这个预训练权重可能更新了，原本的策略模型无法完全收敛，又或者显卡不同，每次更新算法细节不同
                # 反正只要达到指定loss值 0.00030 就差不多完全收敛了
                type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w18',
                # type='Pretrained', checkpoint='/home/xyli/openmmlab_300vw/work_dirs_300vw_v2/td-hm_hrnetv2-w18_8xb64-60e_300w-256x256_2/best_NME_epoch_53.pth'
            ),
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
        out_channels=68, # 每个通道对应68个关键点
        deconv_out_channels=None,
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
# dataset_type = 'Face300WDataset'
# data_mode = 'topdown' 
# data_root = 'data/300w/'
# data_root = 'E:/mmpose/data/300vw'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'), # 这里要标准化bbox(x,y,w,h)

    # 做了变换可能适应于困难样本，但在普通验证集上效果不好
    dict(type='RandomFlip', direction='horizontal'), # 这里要知道图片宽、高
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=60,
        scale_factor=(0.75, 1.25),
        # scale_factor=(0.5, 1.25),
    
    ), # 这里可视化发现框+关键点变换后都蛮准的

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



# 300w dataset
dataset_300w =dict(
    type='Face300WDataset',

    data_root = data_root_300w,

    data_mode='topdown',
    ann_file='annotations/face_landmarks_300w_train.json',
    data_prefix=dict(img='images/'),
    pipeline=train_pipeline,
) 
# 300vw dataset
dataset_300vw =dict(
    type='Face300WDataset',

    data_root = data_root_300vw,
    
    data_mode='topdown',
    ann_file='annotations/train.json',
    data_prefix=dict(img='images/'),
    pipeline=train_pipeline,
) 
# 300vw + 300w
# warning: you should cross out the validate 'meta300w == meta300vw?' at the MMEngine code in system env.
dataset_all = dict(
    type='ConcatDataset',
    datasets=(dataset_300w, dataset_300vw)
)

# validation in 300w
dataset_vali = dict( 
    type='Face300WDataset',
    # data_root=data_root_300w,
    # data_root=data_root_300vw,
    data_root='/home/xyli/data',
    # data_root=data_root_300vw_deblur,
    
    data_mode='topdown',

    # ann_file='annotations/face_landmarks_300w_valid.json', # all the validation data
    # ann_file='annotations/face_landmarks_300w_valid_challenge.json',
    # ann_file='annotations/face_landmarks_300w_valid_common.json',
    # ann_file='annotations/face_landmarks_300w_test.json', # no Test data in server.

    # ann_file='annotations/300VW_blur_label_list_256_test_mmpose.json',
    # ann_file='annotations/300VW_blur_label_list_256_train_mmpose.json',
    # ann_file='annotations/300VW_blur_label_list_256_train_mmpose.json',
    ann_file='546/annot/train.json', 
    # ann_file='data/annotations/300VW_blur_label_list_256_test_mmpose.json',

    # data_prefix=dict(img='images/'),
    data_prefix=dict(img='546/Sharp/'),
    # data_prefix=dict(img='ESTRNN/2024_02_27_08_41_41_ESTRNN_300vw/300vw_ESTRNN_test/'),
    # data_prefix=dict(img='ESTRNN/2024_02_27_14_58_03_ESTRNN_300vw/300vw_ESTRNN_test/'),
    

    test_mode=True,
    pipeline=val_pipeline,
)



# data loaders
train_dataloader = dict(

    batch_size=64,
    num_workers=2,

    # If it is true, GPU will be "out of memory".
    persistent_workers=True, # keep the data in memory all the time, need the num_workers > 0
    # persistent_workers=False,

    sampler=dict(type='DefaultSampler', shuffle=True),

    dataset = dataset_all # 300vw + 300w
    # dataset = dataset_300w # 300w
    # dataset = dataset_300vw # 300vw

    # dataset=dict(
    #     type='Face300VWDataset',
    #     data_root='E:/mmpose/data/300vw',
    #     ann_file='annotations/train.json',
    #     data_prefix=dict(img='images/'),
    #     pipeline=train_pipeline,
    # )

    # dataset=dict(
    #     type='Face300WDataset',
    #     data_root='/home/xyli/data/300vw',
    #     data_mode='topdown',
    #     ann_file='annotations/train.json',
    #     data_prefix=dict(img='images/'),
    #     pipeline=train_pipeline,
    # )

)

# 用300w的验证集验证
val_dataloader = dict(
              
    batch_size=32,
    num_workers=2,

    persistent_workers=True, # need the num_workers > 0
    # persistent_workers=False,

    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset = dataset_vali
)

test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='NME',
    norm_mode='keypoint_distance',
)
test_evaluator = val_evaluator
