import torch
import torch.nn as nn
import torch.nn.init


class MirrorPaddingConv2d(nn.Module):
    """ 
        mirror padding used to replace padding
        
        >>> self.conv = nn.Conv2d(...)
        >>> self.conv = MirrorPaddingConv(...)
    """
    def __init__(self, in_channel, out_channel):
        super(MirrorPaddingConv2d).__init__()
        self.padding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        return self.relu(x)
