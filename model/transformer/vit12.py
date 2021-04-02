from typing import *
import torch.nn as nn
import torch
from model.transformer.attention import Block
from model.transformer.config import Config
from copy import deepcopy


class TransformerLayer(nn.Module):
    def __init__(self):
        """input is (batch, 256, 768), output is (batch, 256, 768)"""
        super(TransformerLayer, self).__init__()
        self.layers = [Block(Config()) for _ in range(12)]
        self.layers = nn.ModuleList(self.layers)

    def get_12layers(self) -> List[Block]:
        return list(deepcopy(self.layers))


def get_vit12() -> List[Block]:
    config = Config()
    assert config.hidden_size == 768 and config.num_heads == 12 and config.mlp_dim == 3072
    model = TransformerLayer()
    path = r"D:\MY\SOD\Tiamat\model\transformer\ViT12.pkl"
    model.load_state_dict(torch.load(path))
    return model.get_12layers()
    

class Transformer12(nn.Module):
    """transformer with 12 layers"""
    def __init__(self):
        super(Transformer12, self).__init__()
        self.layers = get_vit12()
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input size must be (b, 256, 768), and output size is same as input"""
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def test():
        net = Transformer12()
        t = torch.randn(3, 256, 768)
        print(net(t).size())


if __name__ == "__main__":
    Transformer12.test()
