_base_ = [
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='RepVGGplus',
        pretrained='/data/dingxiaohan/model_files/RepVGG-81424-2se_try01spl_custl2tworew_wd4e5_smx_autoweak_nest_lrsEZwm_epoch195_renamed.pth',
        num_blocks=[8,14,24,1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        strides=(2, 2, 1, 1, 1),
        block_groups=None,
        deploy=False,
        use_post_se=False,
        with_cp=True,
        use_custom_L2=True,
        use_pre_se=True),       #TODO note this
    decode_head=dict(
        type='PSPHead',
        in_channels=2560,
        in_index=4,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
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
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='IdentityHead',
            in_index=5, loss_weight=2e-5)
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

optimizer=dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4,
    paramwise_cfg = dict(
        custom_keys={
            'rbr_identity': dict(decay_mult=0.08),       # Because 5e-4 * 0.08 = 4e-5
            'rbr_dense': dict(decay_mult=0),
            'rbr_1x1': dict(decay_mult=0),
        }))

optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)