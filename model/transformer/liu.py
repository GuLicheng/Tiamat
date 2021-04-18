from typing import *
import torch 
import torch.nn as nn
from model.someblocks.resnet_block import ResNetBlock
import torch.nn.functional as F
from model.transformer.vit12 import get_vit12

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = nn.Sequential(
            ResNetBlock(3, 64),
            nn.MaxPool2d(kernel_size=2), # 128
            ResNetBlock(64, 128),
            nn.MaxPool2d(kernel_size=2), # 64
            ResNetBlock(128, 256),
            nn.MaxPool2d(kernel_size=2), # 32
            ResNetBlock(256, 512),
            nn.MaxPool2d(kernel_size=2), # 16
            ResNetBlock(512, 1024)
        )
        self.transformer = get_vit12()
        self.conv1d = nn.Conv1d(1024, 768, kernel_size=1)

    def forward(self, Xs: List[torch.Tensor]):

        size = len(Xs)
        for i in range(size):
            Xs[i] = self.backbone(Xs[i]) 
            Xs[i] = Xs[i].mean(dim=(-1, -2)).view(1, 1, 1024)

        print(Xs[0].size())
        # exit()#

        features = torch.cat(Xs, dim=1)
        features = torch.cat((features, features, features), dim=1)
        features = self.conv1d(features)
        return features
    
    @staticmethod
    def test():
        model = Net()
        t = [torch.randn(1, 3, 256, 256)] * 4
        t = model(t)
        print(t.size())

if __name__ == "__main__":
    Net.test()
    