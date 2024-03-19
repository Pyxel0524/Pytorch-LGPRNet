# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:03:11 2024

LGPRNet implement by pytorch

@author: zpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()
        self.branch1_1 = nn.Conv2d(in_channels, int(out_channels / 4), kernel_size=1)

        self.branch2_1 = nn.Conv2d(in_channels, int(out_channels / 4), kernel_size=1)
        self.branch2_2 = nn.Conv2d(int(out_channels / 4), int(out_channels / 4), kernel_size=3, padding=1)

        self.branch3_1 = nn.Conv2d(in_channels, int(out_channels / 4), kernel_size=1)
        self.branch3_2 = nn.Conv2d(int(out_channels / 4), int(out_channels / 4), kernel_size=3, padding=1)

        self.branch4_2 = nn.Conv2d(in_channels, int(out_channels / 4), kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1_1(x)

        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)

        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)

        branch4 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch4 = self.branch4_2(branch4)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class LGPRNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LGPRNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(11, 3))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=(7, 3))

        self.inception1 = Inception(192, 256)
        self.inception2 = Inception(256, 480)
        self.inception3 = Inception(480, 512)
        self.inception4 = Inception(512, 512)
        self.inception5 = Inception(512, 512)
        self.inception6 = Inception(512, 528)
        self.inception7 = Inception(528, 832)
        self.inception8 = Inception(832, 832)
        self.inception9 = Inception(832, 1024)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(6, 2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 2))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 1))

        self.avgpool = nn.AvgPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(4096, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool3(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.maxpool3(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        out = self.fc(x.view(x.size(0), -1))
        # out = self.sigmoid(x)

        return out


if __name__ == "__main__":
    net = LGPRNet(100, 3)
    test = torch.randn(2, 100, 369, 11)
    out = net(test)
