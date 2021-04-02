import cv2 as cv
import torch
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

def layer(x: int) -> torch.Tensor:
    w = int(math.sqrt(x // 3))
    return torch.arange(x).float().view((1, 3, w, -1))

l1 = layer(12).mean(dim=(-1, -2))
l2 = layer(12).mean(-1).mean(-1)
l3 = layer(12).mean(dim=3).mean(dim=2)
print(l1)
print(l2)
print(l3)



