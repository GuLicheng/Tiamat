from model.transformer.config import Config
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import *
from model.someblocks.resnet_block import ResNetBlock
from model.transformer.vit12 import Transformer12
from torch.nn.modules.sparse import Embedding

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

        self.conv_cat = nn.Conv1d(256*5, 256, 1, 1)

        # batch == 1, no batch_norm

    def forward(self, features: List[torch.Tensor]):
        # extract features from ResNet and Transformer
        # 1.ResNet
        length = len(features)
        for i in range(length):
            features[i] = self._extract_features(features[i])
        
        # reshape (b, 256, w, h) to (b, 256, 768)
        # assume self.embeddings can reshape feature vector to (b, 256, 768)
        for i in range(length):
            features[i] = self.embeddings(features[i])
        
        # 2.Transformer
        for i in range(length):
            features[i] = self.transformer(features[i])
        
        # 5 features with size(b, 256, 768) features
        fusion_feature = torch.cat(features, dim=1)
        fusion_feature = self.conv_cat(fusion_feature)

        # fusion feature with size(b, 256, 768)
        b, c, _ = fusion_feature.size()
        fusion_feature = fusion_feature.mean(dim=-1).view(b, c, 1) # (b, 256, 1)
        for i in range(length):
            features[i] = F.conv1d(features[i], weight=fusion_feature)
        
        



    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """size change: (b, 3, w, h) -> (b, 256, w, h)"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x 


    @staticmethod
    def test():
        model = Net()
        t = [torch.randn(1, 3, 256, 256) for _ in range(5)]
        result = Net(t)
        for res in result:
            print(res.size())
        