
import numpy as np
from base_backbone import BaseBackbone
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from mmcv.cnn import (constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm
import math
from mmcv.runner.base_module import ModuleList

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.FloatTensor(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.FloatTensor(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.adj = nn.Parameter(torch.FloatTensor(12, 12))
        self.adj.data.uniform_(0, 1 / 12)
        self.adj.data = torch.eye(12, 12) + self.adj.data
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        h_shape = h.shape
        output_all = torch.ones((*h_shape[:2], self.out_features)).to(h.device)

        for i in range(h_shape[0]):
            Wh = torch.mm(h[i], self.W.to(h.device))  # h.shape: (N, in_features), Wh.shape: (N, out_features)
            e = self._prepare_attentional_mechanism_input(Wh)

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(self.adj.to(h.device) > 0, self.adj.to(h.device) * e, zero_vec.float())
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh)
            if self.bias is not None:
                h_prime = h_prime + self.bias.to(h.device)
            else:
                h_prime = h_prime
            output_all[i] = h_prime

        if self.concat:
            return F.elu(output_all)
        else:
            return output_all

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid=12, nclass=None, dropout=0.6, alpha=0.2, nheads=3):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = ModuleList()
        for _ in range(nheads):
            self.attentions.append(GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True))
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.elu(self.out_att(x))
        return x


class GcnMain(BaseBackbone):
    def __init__(self,
                 nfeat=64,
                 nhid_layer1=32,
                 nhid_layer2=16,
                 nhid_layer3=8,
                 init_cfg=None):
        super(GcnMain, self).__init__(init_cfg)

        self.gc1 = GAT(nfeat, nhid_layer1, nhid_layer1)
        self.gc2 = GAT(nhid_layer1, nhid_layer2, nhid_layer2)
        self.gc3 = GAT(nhid_layer2, nhid_layer3, 3)
        # self.gc_mali = GAT(nhid_layer2, 32)
        # self.gc_attn = GAT(nhid_layer2, 32)
        self.cls_mali = nn.Sequential(nn.Linear(12 * 3, 2))
        self.cls_attn = nn.Sequential(nn.Linear(12 * 3, 12 * 2))


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
        x1 = self.gc1(x)
        x1 = self.gc2(x1)
        x1 = self.gc3(x1)
        x_mali = self.cls_mali(x1.view(batch_size, -1))
        x_attn = self.cls_attn(x1.view(batch_size, -1)).view(batch_size, -1)
        return x_attn, x_mali

# model = GcnMain()
# x = torch.rand((4, 12, 64))
# y1, y2 = model(x)
# print(y1.size(), y2.size())