from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionBlock(nn.Module):
    
    def __init__(self, cin, cout, dim=1):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=1)
        self.dim = dim

    def forward(self, *args):
        """Fusion n tensors to one tensor via concat and conv"""
        features = torch.cat(args, dim=self.dim)
        x = self.conv(features)
        return x

    @staticmethod
    def test():
        model = FusionBlock(256*5, 256)
        tensors = [torch.randn(1, 256, 16, 16) for _ in range(5)]
        res = model(*tensors)
        print(res.size())

if __name__ == "__main__":
    FusionBlock.test()
