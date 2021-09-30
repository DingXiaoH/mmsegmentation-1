# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from ..builder import LOSSES



@LOSSES.register_module()
class IdentityLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0,
                 loss_name='loss_custom'):
        super(IdentityLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, x, **kwargs):
        return x * self.loss_weight

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
