import torch
import torch.nn as nn

class To3Channel(nn.Module):

    def __init__(self, inc, is_norm = False):
        super().__init__()
        self.inc = inc
    
        self.conv1 = nn.Conv2d(inc, inc, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(inc) if is_norm else nn.Identity()

        self.conv2 = nn.Conv2d(inc, 3, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(3) if is_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)  

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)

        return x
       

    