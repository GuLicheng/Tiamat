import torch 
import torch.nn as nn
from model.someblocks.resnet_block import ResNetBlock
import torch.nn.functional as F

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

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.mean(dim=(-1, -2)).view(1, 1024, 1, 1) # avg pool
        return x
    
    @staticmethod
    def test():
        model = Net()
        t = torch.randn(1, 3, 256, 256)
        t = model(t)
        print(t.size())

if __name__ == "__main__":
    Net.test()
    