from functools import *
from operator import *
from typing import *

import torch.nn as nn


def add_printer_into_sequence(seq: nn.Sequential, debugger,
                              filter_types: Tuple) -> nn.Sequential:
    return nn.Sequential(*list(
        reduce(add, [(m, debugger(m)) for m in filter(lambda x: isinstance(x, filter_types), seq)])))


class Printer1(nn.Module):
    """
        Add this class after each nn.Module and print it's shape for debugging
        Your don't have to know what T is
        Warning:
            this class should .........
    """

    def __init__(self, T):
        super(Printer1, self).__init__()
        self.T = type(T).__name__

    def __call__(self, x):
        """defined your debugger here"""
        print(self.T, end=" ")
        print(x.shape)
        return x


