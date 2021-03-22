from functools import *
from operator import *
from typing import *
import torch
import torch.nn as nn
import sys
sys.path.append("..")
sys.path.append("...")

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


def getLayers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []
 
    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """
 
        # get all layers of the model
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)
 
            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, torch.nn.Module):
                unfoldLayer(module)
 
    unfoldLayer(model)
    for layer in layers:
        print(layer)