import torch
import torch.nn as nn


depth = (1, 1, 3, 3)
n, c, h, w = depth
xx = torch.arange(0, w).view(1, -1).repeat(h, 1).float().cuda().view(1, 1, h, w).repeat(n, 1, 1, 1)
yy = torch.arange(0, h).view(1, -1).repeat(1, w).float().cuda().view(1, 1, h, w).repeat(n, 1, 1, 1)
print(f"xx = {xx}, yy = {yy}")
for i in range(1, 4):
    xy = torch.cat([xx, yy], 1).view(n, 2, -1)
    xx = xy[:, 0, :].view(n, 1, h, w)
    yy = xy[:, 1, :].view(n, 1, h, w)
    print(f"xx' = {xx}, yy' = {yy}")