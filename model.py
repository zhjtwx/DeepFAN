# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
#from functools import partial
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch.autograd import Variable
import pdb

# only add dropout layer in the encoder part
#####################################################
#unet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, upper = 0, drop_rate=0):
        super(Bottleneck, self).__init__()
        self.dropRate = drop_rate
        self.upper = upper
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        if self.upper == 0:
            self.conv2 = nn.Conv3d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            if stride == 1:
                self.conv2 = nn.Conv3d(
                    planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            else:
                self.conv2 = nn.ConvTranspose3d(
                    planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)

        self.bn2 = nn.BatchNorm3d(planes)
        self.dp = nn.Dropout(self.dropRate)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.upper == 0:
            out=self.dp(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class ResNet_UNet3D_decoder_multi_conv(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 block,
                 layers,
                 shortcut_type='B'):
        self.inplanes = 32
        super(ResNet_UNet3D_decoder_multi_conv, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channel,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.kernel_list = [64, 128, 256, 256]

        self.layer1 = self._make_layer(block, self.kernel_list[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, self.kernel_list[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, self.kernel_list[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, self.kernel_list[3], layers[3], shortcut_type, stride=2)

        self.ec8 = self.encoder(self.kernel_list[3] + self.kernel_list[2] * block.expansion, self.kernel_list[2],
                                bias=False, batchnorm=True)
        self.ec9 = self.encoder(self.kernel_list[2], self.kernel_list[2], bias=False, batchnorm=True)
        self.ec10 = self.encoder(self.kernel_list[2] + self.kernel_list[1] * block.expansion, self.kernel_list[1],
                                 bias=False, batchnorm=True)
        self.ec11 = self.encoder(self.kernel_list[1], self.kernel_list[1], bias=False, batchnorm=True)
        self.ec12 = self.encoder(self.kernel_list[1] + self.kernel_list[0] * block.expansion, self.kernel_list[0],
                                 bias=False, batchnorm=True)
        self.ec13 = self.encoder(self.kernel_list[0], 32, bias=False, batchnorm=True)

        self.dc9 = self.decoder(self.kernel_list[3] * block.expansion, self.kernel_list[3], kernel_size=2, stride=2,
                                bias=False)
        self.dc6 = self.decoder(self.kernel_list[2], self.kernel_list[2], kernel_size=2, stride=2, bias=False)
        self.dc3 = self.decoder(self.kernel_list[1], self.kernel_list[1], kernel_size=2, stride=2, bias=False)
        self.dc0 = self.decoder(32, 16, kernel_size=2, stride=2, bias=False)
        #self.conv2 = self.encoder(16+32, 16, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        #self.conv3 = self.encoder(16, out_channel, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        self.conv2 = nn.Conv3d(16 + 32, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        #self.conv3 = nn.Conv3d(16, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        #self.encoder(16+32, out_channel, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=True)
        #
        self.sigmoid = nn.Sigmoid()

        self.output_conv0 = nn.Sequential(
            self.encoder(self.kernel_list[2], 16, bias=True, batchnorm=True),
            nn.Conv3d(16, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
            #self.encoder(16, 1, bias=True, batchnorm=True)  # for 8x8x8
        )
        self.output_conv1 = nn.Sequential(
            self.encoder(self.kernel_list[1], 16, bias=True, batchnorm=True),
            nn.Conv3d(16, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
            #self.encoder(16, 1, bias=True, batchnorm=True)  # for 16x16x16
        )
        self.output_conv2 = nn.Sequential(
            self.encoder(32, 16, bias=True, batchnorm=True),
            nn.Conv3d(16, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
            #self.encoder(16, 1, bias=True, batchnorm=True)  # for 32x32x32
        )

        self.fusion_conv = nn.Conv3d(4*out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, upper=0, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if upper:
                downsample = nn.Sequential(
                    nn.ConvTranspose3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=stride,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, upper=upper))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, upper=upper))

        return nn.Sequential(*layers)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False, act=True):
        if batchnorm:
            if act:
                layer = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.BatchNorm3d(out_channels))
            else:
                layer = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.ReLU())
        return layer

    def forward(self, x):
        pred_list = []
        #print x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        fea_64 = x  # 64  chl:64
        x = self.maxpool(x)
        x = self.layer1(x)  # output shape: 32 chl: 64x4
        fea_32 = x
        x = self.layer2(x)  # output shape: 16 chl: 128x4
        fea_16 = x
        x = self.layer3(x)  # output shape: 8 chl: 256x4
        fea_8 = x
        #print x.size()
        x = self.layer4(x)  # output shape: 4 chl: 512x4

        x = self.dc9(x)  # output shape: 8 chl: 512
        x = torch.cat((x, fea_8), dim=1)
        x = self.ec9(self.ec8(x))  # output shape: 8 chl: 256
        pred_list.append(F.upsample(self.output_conv0(x), scale_factor=8))

        del fea_8
        x = self.dc6(x)  # output shape: 16 chl: 256
        x = torch.cat((x, fea_16), dim=1)  # 128x4 + 256
        x = self.ec11(self.ec10(x))  # output shape: 16 chl: 128
        pred_list.append(F.upsample(self.output_conv1(x), scale_factor=4))

        del fea_16
        x = self.dc3(x)  # output shape: 32 chl: 128
        x = torch.cat((x, fea_32), dim=1)
        x = self.ec13(self.ec12(x))  # output shape: 16 chl: 64
        pred_list.append(F.upsample(self.output_conv2(x), scale_factor=2))

        del fea_32
        x = self.dc0(x)  # output shape: 64 chl: 64
        d1 = torch.cat((x, fea_64), dim=1)  # 64+64
        del fea_64
        d0 = self.conv2(d1)
        #d0 = self.conv3(d1)
        pred_list.append(d0)
        final_pred = self.fusion_conv(torch.cat([pred_list[0], pred_list[1], pred_list[2], pred_list[3]], dim=1))
        pred_list.append(final_pred)

        return pred_list

def resnet50_UNet_half_multi_conv(n_channels, n_classes):
    model = ResNet_UNet3D_decoder_multi_conv(n_channels, n_classes, Bottleneck, [3, 4, 6, 3])
    return model

class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=True)
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=True)
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=True)
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=True)
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=True)
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=True)
        self.ec6 = self.encoder(256, 256, bias=False, batchnorm=True)
        self.ec7 = self.encoder(256, 512, bias=False, batchnorm=True)

        self.ec8 = self.encoder(256 + 512, 256, bias=False, batchnorm=True)
        self.ec9 = self.encoder(256, 256, bias=False, batchnorm=True)
        self.ec10 = self.encoder(128 + 256, 128, bias=False, batchnorm=True)
        self.ec11 = self.encoder(128, 128, bias=False, batchnorm=True)
        self.ec12 = self.encoder(128 + 64, 64, bias=False, batchnorm=True)
        self.ec13 = self.encoder(64, 16, bias=False, batchnorm=True)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        # self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        # self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        # self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = nn.Conv3d(16, n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding, bias=bias),
                nn.ReLU())
        return layer

    def forward(self, x, iterative_=True):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.ec8(d9)
        d7 = self.ec9(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.ec10(d6)
        d4 = self.ec11(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.ec12(d3)
        d1 = self.ec13(d2)
        del d3, d2

        d0 = self.dc0(d1)
        #d0 = self.sigmoid(d0)

        return [d0]
#####################################################################################
#densenet
def n3ddensenet221ClsNew(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = N3DDenseNetClsNew(num_init_features=64, growth_rate=32, block_config=(24, 36, 48), **kwargs)
    return model

class N3DDenseNetClsNew(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(N3DDenseNetClsNew, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features


    def forward(self, x):
        out = self.features(x)
        out = F.relu(out)
        # print(out.size())
        out = F.avg_pool3d(out, kernel_size=3).view(x.size(0), -1)

        out = self.classifier(out)

        return out

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))
#######################################################################
#bcnn
class fandong_net_bcnn(nn.Module):
    def __init__(self, n_channels =1, num_classes =2, pool = 1, dp = 0.5, bias = False, use_sigmoid = True):
        self.in_channel = n_channels
        self.n_classes = num_classes
        self.pool = pool
        self.dp = dp
        self.bias = bias
        self.use_sigmoid = use_sigmoid
        self.features = None
        super(fandong_net_bcnn, self).__init__()

        self.conv1   = self.conv(in_channels=self.in_channel, out_channels=16, bias=True, batchnorm=True)
        self.conv1_2 = self.conv(in_channels=16, out_channels=16, bias=True, batchnorm=True)
        #36*36
        self.pool1 = nn.MaxPool3d(2)
        #18*18

        self.conv2_1 = self.conv(in_channels=16, out_channels=32, kernel_size=2, bias=True, batchnorm=True)
        #19
        self.conv2_2 = self.conv(in_channels=32, out_channels=32, kernel_size=2, padding=0, bias=True, batchnorm=True)
        #18
        self.conv2_3 = self.conv(in_channels=32, out_channels=32, kernel_size=2, bias=True, batchnorm=True)
        #19
        self.conv2_4 = self.conv(in_channels=32, out_channels=32, kernel_size=2, padding=0, bias=True, batchnorm=True)
        #18
        self.pool2 = nn.MaxPool3d(2)
        #9*9
        self.conv3_1 = self.conv(in_channels=32, out_channels=64, kernel_size=2, bias=True, batchnorm=True)
        #10
        self.conv3_2 = self.conv(in_channels=64, out_channels=64, kernel_size=2, padding=0, bias=True, batchnorm=True)
        #9
        self.conv3_3 = self.conv(in_channels=64, out_channels=64, kernel_size=2, bias=True, batchnorm=True)
        #10
        self.conv3_4 = self.conv(in_channels=64, out_channels=64, kernel_size=2, padding=0, bias=True, batchnorm=True)
        #9
        #self.pool3 = nn.MaxPool3d(2)

        #4*4*4*64
        self.classifier = nn.Sequential(
            nn.Linear(64**2, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dp),
            nn.Linear(256, self.n_classes)
        )


    def conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        #x = self.pool3(x)
        cube_size = x.size(2)
        x = x.view(batch_size, 64, cube_size*cube_size*cube_size)
        self.features = x
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (cube_size*cube_size*cube_size)
        x = x.view(batch_size, 64**2)
        x = torch.sqrt(x + 1e-12)
        x = F.normalize(x,2, dim=1)
        x = self.classifier(x)
        return x
###################################
#densenet 1st conv321
def n3ddensenet221ClsNew_1stconv321(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = N3DDenseNetClsNew_1stconv321(num_init_features=64, growth_rate=32, block_config=(24, 36, 48), **kwargs)
    return model

class N3DDenseNetClsNew_1stconv321(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(N3DDenseNetClsNew_1stconv321, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features


    def forward(self, x):
        out = self.features(x)
        out = F.relu(out)
        # print(out.size())
        out = F.avg_pool3d(out, kernel_size=3).view(x.size(0), -1)

        out = self.classifier(out)

        return out
###################################
#densenet 1st conv311
def n3ddensenet221ClsNew_1stconv311(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = N3DDenseNetClsNew_1stconv311(num_init_features=64, growth_rate=32, block_config=(24, 36, 48), **kwargs)
    return model

class N3DDenseNetClsNew_1stconv311(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(N3DDenseNetClsNew_1stconv311, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features


    def forward(self, x):
        out = self.features(x)
        out = F.relu(out)
        #print(out.size())
        out = F.avg_pool3d(out, kernel_size=out.size(2)).view(x.size(0), -1)

        out = self.classifier(out)

        return out

###################################
#densenet 1st conv311
def n3ddensenet72ClsNew_1stconv311(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = N3DDenseNetClsNew72_1stconv311(num_init_features=64, growth_rate=16, block_config=(6, 12, 24), **kwargs)
    return model

class N3DDenseNetClsNew72_1stconv311(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(N3DDenseNetClsNew72_1stconv311, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        #num_features = 6*6*6*508
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features


    def forward(self, x):
        out = self.features(x)
        out = F.relu(out)
        #print(out.size())
        out = F.avg_pool3d(out, out.size(2)).view(x.size(0), -1)

        out = self.classifier(out)

        return out

###################################
#densenet 1st conv311
def Densenet72_bcnn(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = N3Ddensenet72_bcnn(num_init_features=64, growth_rate=16, block_config=(6, 12, 24), **kwargs)
    return model

class N3Ddensenet72_bcnn(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=16, bn_size=4, drop_rate=0, num_classes=1000):

        super(N3Ddensenet72_bcnn, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        num_features = 520**2
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features


    def forward(self, x):
        out = self.features(x)
        out = F.relu(out)
        #print(out.size())
        batch_size, channel_size, cube_size = out.size(0), out.size(1), out.size(2)
        out = out.view(batch_size, channel_size, -1)
        out = torch.bmm(out, torch.transpose(out, 1, 2)) / (cube_size*cube_size*cube_size)
        out = out.view(batch_size, channel_size**2)
        out = torch.sqrt(out + 1e-12)
        out = F.normalize(out,2, dim=1)
        #print out.size()
        out = self.classifier(out)

        return out
    
###################################
#densenet 1st conv311
def Densenet72_cam(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = N3DDenseNetClsNew72_cam(num_init_features=32, growth_rate=16, block_config=(6, 12, 24), **kwargs)
    return model

class N3DDenseNetClsNew72_cam(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(N3DDenseNetClsNew72_cam, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        
        #self.last_conv = nn.Conv3d(num_features, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        
        num_features = 512
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features

    def forward(self, x):
        out = self.features(x)
        #print(out.size())
        out = F.relu(out)
        feat = out
        out = F.avg_pool3d(out, kernel_size=out.size(2)).view(out.size(0), -1)
        #print(out.size())
        out = self.classifier(out)

        return out, feat