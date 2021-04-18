from model.someblocks.resnet_block import ResNetBlock
from typing import *

from torch.nn.functional import pad
import torch.nn as nn
import torch
from model.transformer.config import Config
from model.someblocks.resnet_block import ResNetBlock
from model.saliencymap_generator import SaliencyMapGenerator
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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = ResNetBlock(512, 256)
        self.conv2 = ResNetBlock(256, 128)
        self.conv3 = ResNetBlock(128, 64)
        self.pred = SaliencyMapGenerator(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, features):
        x = self.upsample(x)
        x = self.conv1(x)
        x = x + features[0]
        x = self.relu(x)

        x = self.upsample(x)
        x = self.conv2(x)
        x = x + features[1]
        x = self.relu(x)

        x = self.upsample(x)
        x = self.conv3(x)
        x = x + features[2]
        x = self.relu(x)

        x = self.upsample(x)
        x = self.pred(x)
        return x


if __name__ == "__main__":
    Embedding.test()