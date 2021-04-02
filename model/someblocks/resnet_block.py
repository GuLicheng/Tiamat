import torch 
import torch.nn as nn
from typing import *

class ResNetBlock(nn.Module):
    """Two Layers ResNet Block"""
    def __init__(self, cin: int, cout: int, batch_norm: bool = False, cmid: Optional[int] = None):
        """
            Paras:
                cin: in_channels
                cout: out_channels
                batch_norm: whether use batch norm layer
                cmid: default is cout
        """
        super(ResNetBlock, self).__init__()
        self.cmid = cmid if cmid is not None else cout

        self.conv1 = nn.Conv2d(cin, self.cmid, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(self.cmid) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.cmid, cout, 3, 1, 1)
        self.conv3 = nn.Conv2d(cin, cout, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        y = self.conv3(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.relu(x + y)

    @staticmethod
    def Test():
        t = torch.arange(12).float().view((1, 3, 2, 2))
        blk = ResNetBlock(3, 128, cmid=64)
        res = blk(t)
        print(res.size())

if __name__ == "__main__":
    ResNetBlock.Test()