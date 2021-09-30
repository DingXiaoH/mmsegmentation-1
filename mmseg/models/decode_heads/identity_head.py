# Copyright (c) OpenMMLab. All rights reserved.

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class IdentityHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(IdentityHead, self).__init__(**kwargs)


    def forward(self, inputs):
        """Forward function."""
        return inputs[self.in_index]
