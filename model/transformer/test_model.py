from model.co_saliency.multi_scale import MultiScaleGenerator
from model.transformer.config import Config
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import *
from model.someblocks.resnet_block import ResNetBlock
from model.transformer.vit12 import Transformer12
from model.transformer.embedding import Embedding, Decoder

class Net(nn.Module):
    
    """"""
    def __init__(self):
        super(Net, self).__init__()
        
        is_bn = False # whether use batch_norm

        self.conv1 = ResNetBlock(3, 64, is_bn)
        self.conv2 = ResNetBlock(64, 128, is_bn)
        self.conv3 = ResNetBlock(128, 256, is_bn)

        self.relu_ = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=1)
        self.embeddings = Embedding(Config())
        self.transformer = Transformer12()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.conv_cat = nn.Conv1d(256*5, 256, 1, 1)

        # batch == 1, no batch_norm
        self.conv4 = ResNetBlock(768, 512)  # for change channels
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.multi_scale_generator = MultiScaleGenerator(3, [64, 128, 256], Config())
        self.decoder = Decoder()

    def forward(self, features: torch.Tensor):
        """[1, 3, 3, 256, 256]"""
        features = list(torch.split(features, 1, dim=1))
        # 3 * (1, 3, 256, 256)

        # transform result
        high_level_features = [None] * len(features)
        multi_scale_features = [None] * len(features)
        # extract features from ResNet and Transformer
        # 1.ResNet
        length = len(features)
        for i in range(length):
            high_level_features[i] = features[i].squeeze(0)
            high_level_features[i], multi_scale_features[i] = self._extract_features(high_level_features[i])
            # reshape (b, 256, w, h) to (b, 256, 768)
            # assume self.embeddings can reshape feature vector to (b, 256, 768)
            high_level_features[i] = self.embeddings(high_level_features[i])
            # print(high_level_features[i].size())
        fusion_feature = torch.cat(high_level_features, dim=0)
        _, size, dimension = fusion_feature.size()
        fusion_feature = fusion_feature.mean(dim=0).view(1, size, dimension)
        high_level_features.append(fusion_feature)

        # 2.Transformer
        for i in range(len(high_level_features)):
            high_level_features[i] = self.transformer(high_level_features[i])
            high_level_features[i] = self.conv4(high_level_features[i].permute(0, 2, 1).view(1, -1, 16, 16))
            # print(high_level_features[i].size())


        saliency_map = []
        for i in range(len(high_level_features) - 1):
            saliency_map.append(self.decoder(high_level_features[i] * high_level_features[-1], multi_scale_features[i][::-1]).unsqueeze(1))

        # 3 * (1, 1, 1, 256, 256)
        features = torch.cat(saliency_map, dim=1)
        return features

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """size change: (b, 3, w, h) -> (b, 256, w, h)"""
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x, s1, s2, s3 = self.multi_scale_generator(x)
        t = (s1, s2, s3)
        return x, t


    @staticmethod
    def test():
        model = Net()
        t = torch.randn(1, 3, 3, 256, 256)
        result = model(t)
        for res in result:
            print(res.size())
        
if __name__ == "__main__":
    Net.test()