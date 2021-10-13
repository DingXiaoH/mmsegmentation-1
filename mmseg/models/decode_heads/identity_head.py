# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..builder import HEADS

@HEADS.register_module()
class IdentityHead(BaseModule, metaclass=ABCMeta):


    def __init__(self,
                 in_index,
                 loss_weight,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(IdentityHead, self).__init__(init_cfg)
        self.in_index = in_index
        self.loss_weight = loss_weight

    def extra_repr(self):
        """Extra repr."""
        s = f'in_index={self.in_index} '
        return s

    @auto_fp16()
    def forward(self, inputs):
        return inputs

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        return {'loss_cust_l2': inputs[self.in_index] * self.loss_weight}

    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs)