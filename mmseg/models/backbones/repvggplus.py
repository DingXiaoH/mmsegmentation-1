import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from ..utils.se_layer import SELayer
import torch
from ..builder import BACKBONES

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('relu', nn.ReLU())
    return result

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGplusBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False,
                 use_post_se=False):
        super(RepVGGplusBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()

        if use_post_se:
            self.post_se = SELayer(out_channels, ratio=4)
        else:
            self.post_se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            padding_11 = padding - kernel_size // 2
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)


    def forward(self, x, *args):
        if self.deploy:
            return self.post_se(self.nonlinearity(self.rbr_reparam(x)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        out = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
        out = self.post_se(self.nonlinearity(out))

        if len(args) > 0:      #   Use custom L2. In this case, args[0] should be the accumulated L2
            t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
            t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
            K3 = self.rbr_dense.conv.weight
            K1 = self.rbr_1x1.conv.weight
            l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
            eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
            l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
            return out, args[0] + l2_loss_circle + l2_loss_eq_kernel
        else:
            return out


    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            #   For the 1x1 or 3x3 branch
            kernel, running_mean, running_var, gamma, beta, eps = branch.conv.weight, branch.bn.running_mean, branch.bn.running_var, branch.bn.weight, branch.bn.bias, branch.bn.eps
        else:
            #   For the identity branch
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                #   Construct and store the identity kernel in case it is used multiple times
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(self.in_channels, input_dim, 3, 3)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value.to(branch.weight.device)
            kernel, running_mean, running_var, gamma, beta, eps = self.id_tensor, branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True



class RepVGGplusStage(nn.Module):

    def __init__(self, in_planes, planes, num_blocks, stride, with_cp, use_post_se=False, deploy=False, use_custom_L2=False, block_groups=None):
        super().__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        self.in_planes = in_planes
        for i, stride in enumerate(strides):
            if block_groups is None:
                cur_groups = 1
            else:
                cur_groups = block_groups[i]
            blocks.append(RepVGGplusBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=deploy, use_post_se=use_post_se))
            self.in_planes = planes
        self.blocks = nn.ModuleList(blocks)
        self.with_cp = with_cp
        self.use_custom_L2 = use_custom_L2
        self.deploy = deploy

    def forward(self, x, *args):
        if not self.deploy and self.use_custom_L2:
            L2 = args[0]
            for block in self.blocks:
                if self.with_cp:

                    x, L2 = checkpoint.checkpoint(block, x, L2)
                else:
                    x, L2 = block(x, L2)
            return x, L2
        else:
            for block in self.blocks:
                if self.with_cp:

                    x = checkpoint.checkpoint(block, x)
                else:
                    x = block(x)
            return x

    def switch_to_deploy(self):
        self.deploy = True


@BACKBONES.register_module()
class RepVGGplus(nn.Module):

    def __init__(self,
                 pretrained,
                 num_blocks,
                 width_multiplier,
                 strides=(2,2,2,1,2),
                 block_groups=None,
                 deploy=False,
                 use_post_se=False,
                 with_cp=False,
                 use_custom_L2=False,
                 use_aux_classifiers=False):
        super().__init__()
        self.pretrained = pretrained
        self.deploy = deploy
        self.block_groups = block_groups
        self.use_post_se = use_post_se
        self.with_cp = with_cp
        self.use_custom_L2 = use_custom_L2
        self.use_aux_classifiers = use_aux_classifiers

        if block_groups is None:
            stage1_groups = [1] * num_blocks[0]
            stage2_groups = [1] * num_blocks[1]
            stage3_first_groups = [1] * (num_blocks[2] // 2)
            stage3_second_groups = [1] * (num_blocks[2] - num_blocks[2] // 2)
            stage4_groups = [1] * num_blocks[3]
        else:
            stage1_groups = block_groups[0]
            stage2_groups = block_groups[1]
            stage3_first_groups = block_groups[2][:num_blocks[2] // 2]
            stage3_second_groups = block_groups[2][num_blocks[2] // 2:]
            stage4_groups = block_groups[3]

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGplusBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy,
                                      use_post_se=use_post_se)
        self.stage1 = RepVGGplusStage(self.in_planes, int(64 * width_multiplier[0]), num_blocks[0], stride=strides[0],
                                      with_cp=with_cp, use_post_se=use_post_se, deploy=deploy, use_custom_L2=use_custom_L2, block_groups=stage1_groups)
        self.stage2 = RepVGGplusStage(int(64 * width_multiplier[0]), int(128 * width_multiplier[1]), num_blocks[1], stride=strides[1],
                                      with_cp=with_cp, use_post_se=use_post_se, deploy=deploy, use_custom_L2=use_custom_L2, block_groups=stage2_groups)
        #   split stage3 so that we can insert an auxiliary classifier
        self.stage3_first = RepVGGplusStage(int(128 * width_multiplier[1]), int(256 * width_multiplier[2]), num_blocks[2] // 2, stride=strides[2],
                                            with_cp=with_cp, use_post_se=use_post_se, deploy=deploy, use_custom_L2=use_custom_L2, block_groups=stage3_first_groups)
        self.stage3_second = RepVGGplusStage(int(256 * width_multiplier[2]), int(256 * width_multiplier[2]), num_blocks[2] - num_blocks[2] // 2, stride=strides[3],
                                             with_cp=with_cp, use_post_se=use_post_se, deploy=deploy, use_custom_L2=use_custom_L2, block_groups=stage3_second_groups)
        self.stage4 = RepVGGplusStage(int(256 * width_multiplier[2]), int(512 * width_multiplier[3]), num_blocks[3], stride=strides[4],
                                      with_cp=with_cp, use_post_se=use_post_se, deploy=deploy, use_custom_L2=use_custom_L2, block_groups=stage4_groups)



    def init_weights(self, pretrained=None):
        weights = torch.load(self.pretrained, map_location='cpu')
        if 'model' in weights:
            weights = weights['model']

        renamed_weights = {}
        for k, v in weights.items():
            if 'linear' in k or 'aux' in k:
                continue
            renamed_weights[k.replace('post_se.down', 'post_se.conv1.conv').replace('post_se.up', 'post_se.conv2.conv')] = v

        self.load_state_dict(renamed_weights, strict=True)

    def forward(self, x):

        if self.deploy or (not self.use_aux_classifiers and not self.use_custom_L2):
            result = []
            out = self.stage0(x)
            out = self.stage1(out)
            result.append(out)
            out = self.stage2(out)
            result.append(out)
            out = self.stage3_first(out)
            result.append(out)
            out = self.stage3_second(out)
            result.append(out)
            out = self.stage4(out)
            result.append(out)
            return result

        elif self.use_aux_classifiers and self.use_custom_L2:
            assert 0
            out, L2 = self.stage0(x, 0.0)    # Accumulate the custom L2 value of every block from 0
            out, L2 = self.stage1(out, L2)
            stage1_aux = self.stage1_aux(out)
            out, L2 = self.stage2(out, L2)
            stage2_aux = self.stage2_aux(out)
            out, L2 = self.stage3_first(out, L2)
            stage3_first_aux = self.stage3_first_aux(out)
            out, L2 = self.stage3_second(out, L2)
            out, L2 = self.stage4(out, L2)
            y = self.gap(out)
            y = y.view(y.size(0), -1)
            y = self.linear(y)
            return {
                'main': y,
                'stage1_aux': stage1_aux,
                'stage2_aux': stage2_aux,
                'stage3_first_aux': stage3_first_aux,
                'L2': L2
            }
        elif self.use_aux_classifiers and not self.use_custom_L2:
            assert 0
            out = self.stage0(x)
            out = self.stage1(out)
            stage1_aux = self.stage1_aux(out)
            out = self.stage2(out)
            stage2_aux = self.stage2_aux(out)
            out = self.stage3_first(out)
            stage3_first_aux = self.stage3_first_aux(out)
            out = self.stage3_second(out)
            out = self.stage4(out)
            y = self.gap(out)
            y = y.view(y.size(0), -1)
            y = self.linear(y)
            return {
                'main': y,
                'stage1_aux': stage1_aux,
                'stage2_aux': stage2_aux,
                'stage3_first_aux': stage3_first_aux,
            }
        else:
            result = []
            out, L2 = self.stage0(x, 0.0)    # Accumulate the custom L2 value of every block from 0
            out, L2 = self.stage1(out, L2)
            result.append(out)
            out, L2 = self.stage2(out, L2)
            result.append(out)
            out, L2 = self.stage3_first(out, L2)
            result.append(out)
            out, L2 = self.stage3_second(out, L2)
            result.append(out)
            out, L2 = self.stage4(out, L2)
            result.append(out)
            result.append(L2)
            return result



    def switch_repvggplus_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        if hasattr(self, 'stage1_aux'):
            self.__delattr__('stage1_aux')
        if hasattr(self, 'stage2_aux'):
            self.__delattr__('stage2_aux')
        if hasattr(self, 'stage3_first_aux'):
            self.__delattr__('stage3_first_aux')
        self.deploy = True


#   torch.utils.checkpoint can reduce the memory consumption during training with a minor slowdown. Don't use it if you have sufficient GPU memory.
#   Not sure whether it slows down inference
#   pse for "post SE", which means using SE block after ReLU
def create_RepVGGplus_L2pse(deploy=False, with_cp=False):
    return RepVGGplus(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_post_se=True,
                      with_cp=with_cp)

repvggplus_func_dict = {
'RepVGGplus-L2pse': create_RepVGGplus_L2pse,
}
def get_RepVGGplus_func_by_name(name):
    return repvggplus_func_dict[name]