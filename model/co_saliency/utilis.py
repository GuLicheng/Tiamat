from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_tensorlist_to_tensor(tensors: List[torch.Tensor]):
    """For tensors with size(b, c, w, h), the output will be (b, length, c, w, h)"""
    num = tensors.__len__()
    b, c, w, h = tensors[0].size()
    result = torch.cat(tensors,dim=1).view(b, num, c, w, h)
    return result

def convert_tensor_to_tensorlist(tensor: torch.Tensor):
    """Convert 5-d tensor with size(b, c, num, w, h) to 4-d tensor list"""
    b, c, num, w, h = tensor.size()
    tensors = list(torch.split(tensor=tensor, split_size_or_sections=1, dim=2))
    return tensors

class AveragePool(nn.Module):
    def __init__(self, cin, cout, sample_num):
        super().__init__()
        self.conv = nn.Conv2d(cin * sample_num, cout)

    def forward(self, x: List[torch.Tensor]):
        """(b, c, num, w, h)"""

    def forward1(self, x: List[torch.Tensor]):
        """(b, c, w, h)"""
        features: torch.Tensor = torch.cat(x, dim=1)
        features = self.conv(features)
        weights = features.mean(dim=(-1, -2))
        for i in range(len(x)):
            x[i] = F.conv2d(input=x[i], weight=weights)
        return x


def test1():
    # tensor = torch.randn(8, 3, 256, 256)
    ls = [torch.randn(8, 3, 256, 256) for _ in range(4)]
    result = convert_tensorlist_to_tensor(ls)
    print(result.size())

if __name__ == "__main__":
    test1()
