import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UpsampleBlock(nn.Module):
    """
        >>> low_scalar = torch.randn(8, 256, 12, 12)
        >>> high_scalar = torch.randn(8, 128, 24, 24)
        >>> tensor = UpsampleBlock(256, 128)(low, high)
        >>> tensor.size()
        >>> (8, 128, 24, 24)
    """
    def __init__(self, inc, outc, scale = 2, is_norm = True):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outc) if is_norm else nn.Identity()


    def forward(self, low_scalar, high_scalar):
        x = self.conv(low_scalar)
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=True)
        x = x + high_scalar
        x = self.relu(x)
        x = self.bn(x)
        return x


class SkipConnectBlock(nn.Module):

    """
        >>> high = torch.randn(8, 196, 128)   -> 8, 128, 14, 14
        >>> low = torch.randn(8, 49, 320)     -> 8, 320, 7, 7
        >>> blk = SkipConnectBlock(320, 128, 2)
        >>> res = blk(low, high)
    """
    def __init__(self, inc, outc, scale = 2):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.scale = scale

        self.conv = nn.Conv2d(inc, outc, 1)
        self.relu = nn.ReLU()

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        B, WH, C = low.size()
        low = low.permute(0, 2, 1).view(B, C, int(np.sqrt(WH)), -1).contiguous()
        low = self.conv(low)
        low = F.interpolate(low, scale_factor=self.scale, mode="bilinear", align_corners=True)
        B, WH, C = high.size()
        low = low.view(B, C, -1).permute(0, 2, 1).view(B, WH, C).contiguous()
        return low + high

