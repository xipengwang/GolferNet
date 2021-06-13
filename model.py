#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Pool, Conv, Residual, Hourglass

class CovModel(nn.Module):
    def __init__(self, config={}):
        super(Model, self).__init__()
        num_classes=config.setdefault('num_classes', 10)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class Model(nn.Module):
    def __init__(self, config={}):
        super(Model, self).__init__()
        self.num_stacks=config.setdefault('num_stacks', 1)
        self.num_classes=config.setdefault('num_classes', 10)
        self.inp_dim = 64
        self.out_dim = 128
        self.pre = nn.Sequential(
            Conv(1, 64, kernel_size=3, stride=1, bn=True, relu=True),
            Residual(64, 64),
            Pool(2, 2),
            Residual(64, self.inp_dim)
        )
        self.hgs = nn.ModuleList( [
            nn.Sequential(
                Hourglass(1, self.inp_dim, bn=False, increase=0),
            ) for i in range(self.num_stacks)] )

        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(self.inp_dim, self.inp_dim),
            Conv(self.inp_dim, self.inp_dim, 1, bn=True, relu=True)
        ) for i in range(self.num_stacks)] )

        self.out = nn.Sequential(
            Conv(self.inp_dim, 64, kernel_size=3, stride=1, bn=True, relu=True),
            Residual(64, 64),
            Pool(2, 2),
            Residual(64, self.out_dim)
        )
        self.fc = nn.Linear(7*7*128, self.num_classes)

    def forward(self, x):
        out = self.pre(x)
        for i in range(self.num_stacks):
            hg = self.hgs[i](out)
            out = self.features[i](hg)
        out = self.out(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
