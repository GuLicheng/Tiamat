from typing import *
import torch.nn as nn
import torch

class Config:
    """Configuration for transformer"""
    def __init__(self) -> None:

        """Configurations for self-attention block, Don's touch!"""
        self.num_heads: int = 12  # head of self-attention
        self.hidden_size: int = 768  # hidden_size for full connected layer
        self.mlp_dim = 3072


        self.mlp_act_fn = nn.ReLU(inplace=True)
        self.dropout = 0.1
        self.attention_dropout: float = 0.1  # dropout for self-attention and final output

        self.mode = "train"
        self.is_dropout = self.mode == "train"
        self.image_size = (256, 256)
        self.embedding_inchannels = 3

        self.grid_size: int = (16, 16)
        self.patch_num: int = 12
        self.learning_rate: float = 1e-5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        