_base_ = [
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='RepVGGplus',
        pretrained='/data/dingxiaohan/model_files/NewVGG-81424-2pse_try1wdtworewsgd4e5mm4x256320r15warm00mixup0_best.pth',
        num_blocks=[8,14,24,1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        strides=(2, 2, 2, 1, 2),
        block_groups=None,
        deploy=False,
        use_post_se=True,
        with_cp=True,
        use_custom_L2=False),
    decode_head=dict(
        type='UPerHead',
        in_channels=[160, 320, 640, 2560],
        in_index=[0, 1, 2, 4],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True, #TODO because 769x769
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=640,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=True,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
        dict(
            type='FCNHead',
            in_channels=640,
            in_index=3,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=True,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2))
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

optimizer=dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4,
    paramwise_cfg = dict(
        custom_keys={
            'rbr_identity.weight': dict(decay_mult=1)
        }))

optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)