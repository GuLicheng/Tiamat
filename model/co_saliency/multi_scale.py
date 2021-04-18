from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.someblocks.resnet_block import ResNetBlock
from model.transformer.config import Config

class MultiScaleGenerator(nn.Module):
    """Create three-scalar features map"""
    def __init__(self, cin: int, couts: List[int], config: Config):
        """
            For input:(b, c, w, h), the output is(b, c1, w//2, h//2), (b, c2, w//4, h//4), (b, c3, w//8, h//8)
            Paras:
                couts: output channels for three-scalar
        """
        super(MultiScaleGenerator, self).__init__()
        self.res1 = ResNetBlock(cin, couts[0])
        self.res2 = ResNetBlock(couts[0], couts[1])
        self.res3 = ResNetBlock(couts[1], couts[2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)  # maybe unused

    def forward(self, x: torch.Tensor):
        feature0 = self.res1(x)
        feature1 = self.maxpool1(feature0)
        feature0 = self.res2(feature0)
        feature2 = self.maxpool1(self.res2(feature1))
        feature0 = self.res3(feature0)
        feature3 = self.maxpool1(self.res3(feature2))
        return feature0, feature1, feature2, feature3

    @staticmethod
    def test():
        model = MultiScaleGenerator(3, [64, 128, 256], Config())
        t = torch.randn(1, 3, 256, 256)
        (f0, f1, f2, f3) = model(t)
        print(f0.size(), f1.size(), f2.size(), f3.size())


if __name__ == "__main__":
    MultiScaleGenerator.test()