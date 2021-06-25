#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Pool, Conv, Residual, Hourglass
from loss import HeatmapLoss, FocalLoss

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class Model(nn.Module):
    def __init__(self, config={}):
        super(Model, self).__init__()
        self.num_stacks=config.setdefault('num_stacks', 1)
        # image size = 512 x 512
        inp_dim = config.setdefault('inp_dim', 256)
        out_dim = config.setdefault('out_dim', 18)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.pre = nn.Sequential(
            Conv(3, 64, 7, stride=2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        # feature size = 128 x 128
        n = config.setdefault('hourglass_components', 4)
        n = 4 # 128->64(n=1)->32(n=2)->16(n=3)->8(n=1)
        self.hgs = nn.ModuleList( [
            nn.Sequential(
                Hourglass(n, self.inp_dim, bn=False, increase=0),
            ) for i in range(self.num_stacks)] )

        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(self.inp_dim, self.inp_dim),
            Conv(self.inp_dim, self.inp_dim, 1, bn=True, relu=True)
        ) for i in range(self.num_stacks)] )

        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(self.num_stacks-1)] )
        self.merge_preds = nn.ModuleList( [Merge(out_dim, inp_dim) for i in range(self.num_stacks-1)] )

        self.outs = nn.ModuleList( [Conv(inp_dim, out_dim, 1, relu=False, bn=False) for i in range(self.num_stacks)] )
        # self.loss_func = HeatmapLoss()
        self.loss_func = FocalLoss()

    def forward(self, x):
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.num_stacks):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            prob = preds.sigmoid()
            combined_hm_preds.append(prob)
            if i < self.num_stacks - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        N, nstack, nclass, W, H= combined_hm_preds.shape
        combined_loss = []
        for i in range(self.num_stacks):
            loss = self.loss_func(combined_hm_preds[:, i, :, :], heatmaps)
            combined_loss.append(loss)
        combined_loss = torch.stack(combined_loss, dim=0)
        return combined_loss.mean()
