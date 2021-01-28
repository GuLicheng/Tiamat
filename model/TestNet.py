"""
    Time: 2020/11/11

    This is a simple model for study and we can improve it in someday
"""

import torch.nn as nn


class TestNet(nn.Module):

    def __init__(self, in_channel, out_channel, stride, resize, downsampling_rate=2):
        super(TestNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.resize = resize
        self.downsampling_rate = downsampling_rate

    def forward(self, x):
        pass
