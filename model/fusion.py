import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./model')
from base_backbone import BaseBackbone
from vit import VisionTransformerMul3DMB
from fine_grained import WSDAN_CAL_MULI
from gcn import GcnMain
from mmcv.runner.base_module import ModuleList


class FusionModel(BaseBackbone):
    def __init__(self, vit_cfg={"kernel_size": (1, 1, 1),
                                "in_channels": 1,
                                "strides": ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                                "out_indices": -1,
                                "qkv_bias": True,
                                "drop_rate": 0.0,
                                "drop_path_rate": 0.3,
                                "patch_cfg": dict(input_size=(128, 128, 128)),
                                "init_cfg": dict(
                                    type='Pretrained',
                                    checkpoint='')
                                },
                 fine_grained_cfg={"in_channels": 1,
                                   "num_classes1": 2,
                                   "num_classes2": 2,
                                   "num_classes3": 3,
                                   "remove_aa_jit": False,
                                   "return_feat": True,
                                   "M": 8,
                                   "net": 'trresnet_large',
                                   "init_cfg": None},
                 gcn_cfg={"nfeat": 64,
                          "nhid_layer1": 32,
                          "nhid_layer2": 16,
                          "nhid_layer3": 8,
                          "init_cfg": None},
                 init_cfg=None):
        super(FusionModel, self).__init__(init_cfg)

        self.vit = VisionTransformerMul3DMB(kernel_size=vit_cfg['kernel_size'],
                                            in_channels=vit_cfg['in_channels'],
                                            strides=vit_cfg['strides'],
                                            out_indices=vit_cfg['out_indices'],
                                            qkv_bias=vit_cfg['qkv_bias'],
                                            drop_rate=vit_cfg['drop_rate'],
                                            drop_path_rate=vit_cfg['drop_path_rate'],
                                            patch_cfg=vit_cfg['patch_cfg'],
                                            init_cfg=vit_cfg['init_cfg'],
                                            single_model=False)
        self.fine_grained = WSDAN_CAL_MULI(
                                            in_channels=fine_grained_cfg['in_channels'],
                                            num_classes1=fine_grained_cfg['num_classes1'],
                                            num_classes2=fine_grained_cfg['num_classes2'],
                                            num_classes3=fine_grained_cfg['num_classes3']
                                            )

        self.gcn = GcnMain(nfeat=gcn_cfg['nfeat'],
                           nhid_layer1=gcn_cfg['nhid_layer1'],
                           nhid_layer2=gcn_cfg['nhid_layer2'],
                           nhid_layer3=gcn_cfg['nhid_layer3'],
                           init_cfg=gcn_cfg['nhid_layer3'])

        self.fc_all = nn.Linear(12 * 64, 2)

    def init_weights(self):
        super(GcnMain, self).init_weights()
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
        batch_size = x.size(0)
        vit_cls_mb, vit_nodes = self.vit(x)
        hf0, hf1, hf2, h0, h1, h2 = self.fine_grained(x)
        fine_grained_nodes = [h0, h1, h2]
        all_nodes = torch.cat((vit_nodes + fine_grained_nodes)).view(batch_size, 12, 64)
        all_cls_mb = self.fc_all(all_nodes.view(batch_size, -1))
        x_attn, gcn_cla_mb = self.gcn(all_nodes)
        return gcn_cla_mb, vit_cls_mb, all_cls_mb, hf0, hf1, hf2


# model = FusionModel()
# x = torch.rand((2, 1, 128, 128, 128))
# gcn_cla_mb, vit_cls_mb, all_cls_mb, hf0, hf1, hf2 = model(x)
# print(gcn_cla_mb.size(), vit_cls_mb.size(), all_cls_mb.size(), hf0.size(), hf1.size(), hf2.size())
