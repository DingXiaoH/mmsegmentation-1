# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='RepVGGplus',
        pretrained='/data/dingxiaohan/model_files/NewVGG-81424-2pse_try1wdtworewsgd4e5mm4x256320r15warm00mixup0_best.pth',
        num_blocks=[8,14,24,1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        strides=(2, 2, 1, 1, 1),
        block_groups=None,
        deploy=False,
        use_post_se=True,
        with_cp=True,
        use_custom_L2=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2560,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=640,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
