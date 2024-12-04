import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.modules.module import Module
from collections import OrderedDict
from inplace_abn import InPlaceABN, InPlaceABNSync
import random
from mmcv.cnn import (constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
if False:
    Batchnorm = InPlaceABNSync
else:
    Batchnorm = InPlaceABN
import math
from abc import ABCMeta, abstractmethod
from base_backbone import BaseBackbone

class ABN(nn.Module):
    """Activated Batch Normalization
    This gathers a BatchNorm and an activation function in a single module
    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    activation : str
        Name of the activation functions, one of: `relu`, `leaky_relu`, `elu` or `identity`.
    activation_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu",
                 activation_param=1e-6):
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.activation_param = activation_param
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                         self.training, self.momentum, self.eps)

        if self.activation == "relu":
            return F.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return F.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError("Unknown activation function {}".format(self.activation))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Post-Pytorch 1.0 models using standard BatchNorm have a "num_batches_tracked" parameter that we need to ignore
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(ABN, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                               error_msgs, unexpected_keys)

    def extra_repr(self):
        rep = '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
        if self.activation in ["leaky_relu", "elu"]:
            rep += '[{activation_param}]'
        return rep.format(**self.__dict__)

EPSILON = 1e-6


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool3d(1)

    def forward(self, features, attentions):
        B, C, D, H, W = features.size()
        _, M, AD, AH, AW = attentions.size()

        # match size
        if AD != D or AH != H or AW != W:
            attentions = F.interpolate(attentions, size=(D, H, W), mode='trilinear')

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjkl,injkl->imn', (attentions, features)) / float(D * H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)
        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjkl,injkl->imn', (fake_att, features)) / float(D * H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


def BasicConv3d_ABN(ni, nf, stride=1, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    activation_param = 1e-6
    return nn.Sequential(
        nn.Conv3d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        Batchnorm(num_features=nf, activation=activation, activation_param=activation_param)
    )


# ADL attention
class ADL(nn.Module):
    def __init__(self, drop_prob=0.25, gamma=0.9):#0.125 0.95 ||| 0.25 0.9
        super(ADL, self).__init__()
        self.drop_prob = drop_prob
        self.gamma = gamma
        self.attentions = BasicConv3d_ABN(512, 4, kernel_size=1)

    def forward(self, x):
        if self.training:
            attention_map = x.mean(dim=1, keepdim=True)
            if torch.rand(()).item() < self.drop_prob:
                max_inten = attention_map.flatten(start_dim=1).max(dim=1)[0].view(x.size(0), 1, 1, 1, 1)
                keep = (attention_map < max_inten * self.gamma).type(x.dtype)  # no gradient
            else:
                keep = attention_map.sigmoid()  # mask can have gradients!
            return keep*self.attentions(x)
        else:
            attention_map = x.mean(dim=1, keepdim=True)
            keep = attention_map.sigmoid()
            return keep*self.attentions(x)


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def conv3d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    activation_param = 1e-6
    return nn.Sequential(
        nn.Conv3d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        Batchnorm(num_features=nf, activation=activation, activation_param=activation_param)
    )


class FastGlobalAvgPool3d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool3d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1, 1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastGlobalAvgPool3d()
        self.fc1 = nn.Conv3d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv3d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        # self.activation = hard_sigmoid(inplace=inplace)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=[2, 2, 2]):  # D H W
        super().__init__()
        # assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, D, H, W = x.size()
        x = x.view(N, C, D // self.bs[0], self.bs[0], H // self.bs[1], self.bs[1], W // self.bs[2],
                   self.bs[2])  # (N, C, D//bs, bs, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous()  # (N, bs, bs, bs, C, D//bs, H//bs, W//bs)
        x = x.view(N, C * (self.bs[0] * self.bs[1] * self.bs[2]), D // self.bs[0], H // self.bs[1],
                   W // self.bs[2])  # (N, C*bs^3, D//bs, H//bs, W//bs)
        return x


@torch.jit.script
class SpaceToDepthJit(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, D, H, W = x.size()
        # if config.final_size[2] % 2 ==1:
        block_size = [2, 2, 2]
        # block_size = [1, 2, 2]
        # else:
        #     block_size = [2, 2, 2]
        x = x.view(N, C, D // block_size[0], block_size[0], H // block_size[1], block_size[1], W // block_size[2],
                   block_size[2])  # (N, C, D//bs, bs, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous()  # (N, bs, bs, bs, C, D//bs, H//bs, W//bs)
        x = x.view(N, C * block_size[0] * block_size[1] * block_size[2], D // block_size[0], H // block_size[1],
                   W // block_size[2])  # (N, C*bs^3, D//bs, H//bs, W//bs)
        return x


class SpaceToDepthModule(nn.Module):
    def __init__(self, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class AntiAliasDownsampleLayer(nn.Module):
    def __init__(self, remove_aa_jit: bool = False, filt_size: int = 3, stride: int = 2,
                 channels: int = 0):
        super(AntiAliasDownsampleLayer, self).__init__()

        if not remove_aa_jit:
            self.op = DownsampleJIT(filt_size, stride, channels)
        else:
            self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)


class Downsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels

        assert self.filt_size == 3
        a = torch.tensor([1., 2., 1.])

        # filt = (a[:, None] * a[None, :])
        filt = (a.view(-1, 1, 1) * a.view(1, -1, 1) * a.view(1, 1, -1))
        filt = filt / torch.sum(filt)

        # self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        # self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1, 1, 1), 'replicate')
        return F.conv3d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


@torch.jit.script
class DownsampleJIT(object):
    def __init__(self, filt_size: int = 3, stride: int = 2, channels: int = 0):
        self.stride = stride
        self.filt_size = filt_size
        self.channels = channels

        assert self.filt_size == 3
        assert stride == 2
        a = torch.tensor([1., 2., 1.])
        # filt = (a[:, None] * a[None, :]).clone().detach()
        filt = (a.view(-1, 1, 1) * a.view(1, -1, 1) * a.view(1, 1, -1)).clone().detach()
        filt = filt / torch.sum(filt)
        # self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1)).cuda().half()
        self.filt = filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)).cuda().half()

    def __call__(self, input: torch.Tensor):
        if input.dtype != self.filt.dtype:
            self.filt = self.filt.float()
        input_pad = F.pad(input, (1, 1, 1, 1, 1, 1), 'replicate')

        # return F.conv3d(input_pad, self.filt.cpu(), stride=2, padding=0, groups=input.shape[1])

        return F.conv3d(input_pad, self.filt.cuda(), stride=2, padding=0, groups=input.shape[1])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv3d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv3d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(conv3d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv3d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if stride == 2:
            self.downsample_anti_alias_layer = anti_alias_layer(channels=planes, filt_size=3, stride=2)
        else:
            self.downsample_anti_alias_layer = None

        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
            if self.downsample_anti_alias_layer:
                residual = self.downsample_anti_alias_layer(residual)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = conv3d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv3d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(conv3d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv3d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if stride == 2:
            self.downsample_anti_alias_layer = anti_alias_layer(channels=planes * self.expansion, filt_size=3, stride=2)
        else:
            self.downsample_anti_alias_layer = None
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
            if self.downsample_anti_alias_layer:
                residual = self.downsample_anti_alias_layer(residual)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(nn.Module):

    def __init__(self, layers, in_chans=1, width_factor=1.0, remove_aa_jit=False, return_feat=False):
        super(TResNet, self).__init__()
        self.return_feat = return_feat
        # JIT layers
        self.space_to_depth = SpaceToDepthModule(remove_model_jit=remove_aa_jit)
        anti_alias_layer = partial(AntiAliasDownsampleLayer, remove_aa_jit=remove_aa_jit)
        global_pool_layer = FastGlobalAvgPool3d(flatten=True)

        # TResnet stages
        self.inplanes = int(int(64 * width_factor + 4) / 8) * 8
        self.planes = int(int(64 * width_factor + 4) / 8) * 8
        self.conv1 = conv3d_ABN(in_chans * 2 * 2 * 2, self.planes, stride=1, kernel_size=3)
        self.layer1 = self._make_layer(Bottleneck, self.planes, layers[0], stride=1, use_se=True,
                                       anti_alias_layer=anti_alias_layer)  # 56x56
        self.layer2 = self._make_layer(Bottleneck, self.planes * 2, layers[1], stride=2, use_se=True,
                                       anti_alias_layer=anti_alias_layer)  # 28x28
        self.layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                       anti_alias_layer=anti_alias_layer)  # 14x14

        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', self.space_to_depth),
            ('conv1', self.conv1),
            ('layer1', self.layer1),
            ('layer2', self.layer2),
            ('layer3', self.layer3)]))
        # ('layer4', layer4)]))

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 4) * Bottleneck.expansion
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):  # or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            layers += [conv3d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        return x


class WSDAN_CAL_MULI(BaseBackbone):
    def __init__(self,
                 in_channels=1,
                 num_classes1=2,
                 num_classes2=2,
                 num_classes3=3,
                 remove_aa_jit=False,
                 return_feat=True,
                 M=4,
                 net='trresnet_large',
                 init_cfg=None):
        super(WSDAN_CAL_MULI, self).__init__(init_cfg)
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3
        self.M = M
        self.net = net

        # Network Initialization

        if 'trresnet_large' in net:
            self.features = TResNet(layers=[6, 9, 12], in_chans=in_channels, width_factor=0.5,
                                    remove_aa_jit=remove_aa_jit, return_feat=return_feat)
            self.num_features = self.features.num_features
        else:
            raise ValueError('Unsupported net: %s' % net)

        # Attention Maps
        self.attentions = ADL()
        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # # Classification Layer
        self.fc1 = nn.Linear(self.M * self.num_features, 64, bias=False)
        self.fc2 = nn.Linear(self.M * self.num_features, 64, bias=False)
        self.fc3 = nn.Linear(self.M * self.num_features, 64, bias=False)
        self.fc1_1 = nn.Linear(self.M * self.num_features, 64, bias=False)
        self.fc2_1 = nn.Linear(self.M * self.num_features, 64, bias=False)
        self.fc3_1 = nn.Linear(self.M * self.num_features, 64, bias=False)

        self.fc1_f = nn.Linear(64, self.num_classes1, bias=False)
        self.fc2_f = nn.Linear(64, self.num_classes2, bias=False)
        self.fc3_f = nn.Linear(64, self.num_classes3, bias=False)
        self.fc1_1_f = nn.Linear(64, self.num_classes1, bias=False)
        self.fc2_1_f = nn.Linear(64, self.num_classes2, bias=False)
        self.fc3_1_f = nn.Linear(64, self.num_classes3, bias=False)
        # self.feature_center1 = nn.Parameter(torch.zeros(self.num_classes1, M * self.num_features), requires_grad=False)
        #
        # self.fc2 = nn.Linear(self.M * self.num_features, self.num_classes2, bias=False)
        # self.feature_center2 = nn.Parameter(torch.zeros(self.num_classes2, M * self.num_features), requires_grad=False)
        #
        # self.fc3 = nn.Linear(self.M * self.num_features, self.num_classes3, bias=False)
        # self.feature_center3 = nn.Parameter(torch.zeros(self.num_classes3, M * self.num_features), requires_grad=False)

        self.beta = 5e-2


    def visualize(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)

        # attention_maps = self.attentions(feature_maps)
        attention_maps = torch.abs(self.attentions(feature_maps)).clamp(min=0.0)  # BCDHW

        feature_matrix, _ = self.bap(feature_maps, attention_maps)
        p = self.fc(feature_matrix * 100.)

        return p, attention_maps

    def init_weights(self):
        super(WSDAN_CAL_MULI, self).init_weights()
        if self.init_cfg is not None:
            pretrained = self.init_cfg.get('checkpoint', None)
        else:
            pretrained = None
        if pretrained is not None:
            pretrained = pretrained[0]

        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x):
        feature_maps = self.features(x)
        attention_maps = torch.abs(self.attentions(feature_maps)).clamp(min=0.0)  # BCDHW
        feature_matrix, counterfactual_feature = self.bap(feature_maps, attention_maps)
        # print(feature_matrix.size())
        h0 = self.fc1(feature_matrix)
        h0_ = self.fc1_1(counterfactual_feature)
        h1 = self.fc2(feature_matrix)
        h1_ = self.fc2_1(counterfactual_feature)
        h2 = self.fc3(feature_matrix)
        h2_ = self.fc3_1(counterfactual_feature)
        hf0 = self.fc1_f(h0) - self.fc1_1_f(h0_)
        hf1 = self.fc2_f(h1) - self.fc2_1_f(h1_)
        hf2 = self.fc3_f(h2) - self.fc3_1_f(h2_)

        return hf0, hf1, hf2, h0 - h0_, h1 - h1_, h2 - h2_

#
# def cal_feature():
#     # pretrained = '/mnt/LungLocalNFS/tanweixiong/project/swin_mul_xiehe/work_dirs0314/twx_xie_swin/tr_resnet_cal/best_auc_mean_epoch_53_val.pth'
#     model = WSDAN_CAL_MULI(
#             in_channels=1,
#             num_classes1=2,
#             num_classes2=2,
#             num_classes3=3
#         )
#     return model
# #
# model = cal_feature()
# x = torch.rand((4, 1, 64, 64, 64))
# y1, y2,_,_,_,y5 = model(x)
# print(y1.size(), y2.size(),y5.size())

