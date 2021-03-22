import torch.nn as nn
import torch

import sys
sys.path.append("..")

from model.downsample import DownSample
from model.identity import Identity

class Inception(nn.Module):
    """
        An user-defined Inception, which not change the size of input
        卷积之后，如果要接BN操作，最好是不设置偏置，因为不起作用，而且占显卡内存。
        Replace simple conv2d(3*3) layer
    """
    def __init__(self, in_channels, out_channels, batch_normol=True):
        assert out_channels >= 32 , "in_channels must greater than 32"
        assert out_channels % 4 == 0, "in_channels must be divided by 4"

        super(Inception, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels // 4
        self.bias = not batch_normol

        # first part
        self.conv1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=1, stride=1, bias=self.bias)

        # second part
        self.conv2_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=1, stride=1, bias=self.bias)
        self.conv2_2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=1, bias=self.bias)
        
        # third part
        self.conv3_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=1, stride=1, bias=self.bias)
        self.conv3_2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.conv3_3 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=1, bias=self.bias)

        # forth part
        self.conv4_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=1, stride=1, bias=self.bias)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(self.mid_channels) if batch_normol == True else Identity()
        self.relu = nn.ReLU() 

        # adjust channels for residual 
        self.conv0 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        """
            concat x1, x2, x3, x4
        """
        x1 = self.conv1_1(x)

        x2 = self.conv2_1(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bn(x2)
        x2 = self.relu(x2)

        x3 = self.conv3_1(x)
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        x3 = self.conv3_2(x3)
        x3 = self.bn(x3)
        x3 = self.relu(x3)
        x3 = self.conv3_3(x3)
        x3 = self.bn(x3)
        x3 = self.relu(x3)

        x4 = self.conv4_1(x)
        x4 = self.bn(x4)
        x4 = self.relu(x4)
        x4 = self.pool(x4)

        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv0(x)
        return self.relu(x_cat + x)

class Printer(nn.Module):
    def __init__(self, msg=""):
        super(Printer, self).__init__()
        self.msg = msg
    def forward(self, x):
        print(self.msg, x.size())
        return x

class InceptionNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(InceptionNet, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
            Inception(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Inception(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            DownSample(256), # 112 * 112
            Inception(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            DownSample(512), # 56 * 56
            Inception(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4),  # 224
            nn.Conv2d(512, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    t = torch.randn(8, 64, 224, 224)
    model = Inception(64, 128)
    print(t.size())
    t = model(t)
    print(t.size())