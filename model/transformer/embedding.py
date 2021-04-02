from model.someblocks.resnet_block import ResNetBlock
from typing import *

from torch.nn.functional import pad
import torch.nn as nn
import torch
from model.transformer.config import Config


class Embedding(nn.Module):
    def __init__(self, config: Config):
        super(Embedding, self).__init__()
        self.path_size = config.grid_size  # (16, 16)
        self.image_size = config.image_size  # (256, 256)

        in_channels = 256 # follow with ResNetBlock(x, 256)
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=config.hidden_size, 
            kernel_size=self.path_size[0],  # 16
            stride=self.path_size[0])

        n_patches = (self.image_size[0] // self.path_size[0]) * (self.image_size[1] // self.path_size[1]) # 256
        self.positions_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))  # (1, 256, 768)
        self.is_dropout = config.is_dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        """for input size(b, 256, w, h) -> output size is(b, 256, 768)"""
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x.transpose_(-1, -2)
        embeddings = x + self.positions_embeddings
        if self.is_dropout:
            embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def test():
        model = Embedding(Config())
        t = torch.randn(1, 256, 256, 256)
        t = model(t)
        print(t.size())

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
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor):
        feature0 = self.res1(x)
        feature1 = self.pool(feature0)
        feature0 = self.res2(feature0)
        feature2 = self.pool(feature0)
        feature0 = self.res3(feature0)
        feature3 = self.pool(feature0)
        return feature0, feature1, feature2, feature3

    @staticmethod
    def test():
        model = MultiScaleGenerator(3, [64, 128, 256], Config())
        t = torch.randn(1, 3, 256, 256)
        (f0, f1, f2, f3) = model(t)
        print(f0.size(), f1.size(), f2.size(), f3.size())


if __name__ == "__main__":
    Embedding.test()
    MultiScaleGenerator.test()