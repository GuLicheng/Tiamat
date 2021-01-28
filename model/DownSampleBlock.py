import torch.nn as nn
from math import comb
import torch
import numpy as np
import torch.nn.functional as F

class DownSample(nn.Module):

    def __init__(self, filter_size, stride, channels):
        super(DownSample, self).__init__()

        self.filter_size = filter_size
        self.stride = stride
        self.channels = channels
        self.filter = self.create_filter()
        self.filter = F.softmax(self.filter, dim=0)

        print(self.filter)

    def create_filter(self):
        n = int(self.filter_size)
        # combination
        arr = np.array([comb(n - 1, __i) for __i in range(n)])
        return torch.Tensor(arr[:, None] * arr[None, :])


if __name__ == '__main__':

    for i in range(1, 5):
        d = DownSample(i, 2, 2)
