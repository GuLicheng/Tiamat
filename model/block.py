import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d


class MirrorPadding(nn.Module):
    """ 
        mirror padding used to replace padding
    """
    def __init__(self, in_channel, out_channel):
        super(MirrorPadding).__init__()
        self.padding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        return self.relu(x)

class SelfAttentionBlock(nn.Module):
    """
        A simple self-attention block, which would not change the size of image
    """
    def __init__(self, in_channels, out_channels):
        super(SelfAttentionBlock, self).__init__()

        # basic parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.middle_channels = in_channels // 2

        # Q, K, V matrix
        self.node_q = self._conv3x3()
        self.node_k = self._conv3x3()
        self.node_v = self._conv3x3()

        self.softmax = nn.Softmax(dim=2) # channel

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=self.middle_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # b, c, h, w = x.size()
        node_q = self.node_q(x)
        node_k = self.node_k(x)
        node_v = self.node_v(x)

        b, c, h, w = node_k.size()

        # transpose
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)


    def _conv3x3(self):
        """
            generate a simple 3x3 convolution
        """
        return nn.Conv2d(in_channels=self.in_channels, out_channels=self.middle_channels, kernel_size=1, stride=1, bias=False)
