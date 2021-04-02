import torch
import torch.nn as nn
from model.transformer.config import *
import math


class Attention(nn.Module):
    """Attention block"""
    def __init__(self, config: Config):
        """
            Paras(all from config):
                hidden_size: the num of hidden neural
                num_heads: the num of heads of your self-attention
                attention_dropout: dropout of self-attention output
        """
        super(Attention, self).__init__()

        assert config.hidden_size % config.num_heads == 0, "result must be integer"
        assert 0 < config.attention_dropout < 1, "dropout must between 0 and 1"

        self.is_dropout = config.is_dropout

        """For 12 heads, hidden_size = 768, attention_size = 64"""
        self.num_attention_heads = config.num_heads
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        self.all_head_size = config.hidden_size

        """Use linear to change dimension"""
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(config.attention_dropout)  # dropout for attention
        self.proj_dropout = nn.Dropout(config.attention_dropout)  # dropout for final output

        self.softmax = nn.Softmax(dim=1)  # softmax(Q * K^T / (d_k^1/2)) * V
        self.out = nn.Linear(self.all_head_size, config.hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
            Reshape tensor from 3d to 4d to adjust 2-d image
            For x(b, c, n): 
            new_shape = (b, c, num_attention_heads, self.attention_head_size) -> (b, c, 12, 12)
            (1, 256, 768) -> (1, 12, 256, 64)
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3) 

    def forward(self, x: torch.Tensor):
        # Create Q, K, V matrix
        mix_query_layer = self.query(x)
        mix_key_layer = self.key(x)
        mix_value_layer = self.value(x)
        
        # Reshape matrix
        query_layer = self.transpose_for_scores(mix_query_layer)
        key_layer = self.transpose_for_scores(mix_key_layer)
        value_layer = self.transpose_for_scores(mix_value_layer)

        # attention = softmax(Q * K^T) / (d_k)^1/2
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # (d_k)^1/2
        attention_scores = self.softmax(attention_scores)
        if self.is_dropout:
            attention_scores = self.attn_dropout(attention_scores)

        # attention * V
        context_layer = torch.matmul(attention_scores, value_layer).permute(0, 2, 1, 3).contiguous()
        new_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )  
        context_layer = context_layer.view(*new_layer_shape)  # adjust shape for linear
        # output, just like a conv with same in_channels and out_channels
        attention_output = self.out(context_layer)  
        if self.is_dropout:
            attention_output = self.proj_dropout(attention_output) # dropout

        return attention_output

    @staticmethod
    def test_input():
        attn = Attention(Config())
        t = torch.randn(1, 256, 768)
        print(f"Attention test: {attn(t).size()}")

class Mlp(nn.Module):
    """Follow with Attention Block"""
    def __init__(self, config: Config):
        """
            Paras(all from config):
                hidden_size: same as above
                mil_dim: midden dimension between two linear layers
                mlp_act_fn: activate function for linear
                dropout: same as above
        """
        super(Mlp, self).__init__()
        # We have to feedback the result to the next Mlp, 
        # so the output dimension must same as the input dimension(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = config.mlp_act_fn  # for standard transformer,it is relu
        self.dropout = nn.Dropout(config.dropout)  
        self.is_dropout = config.is_dropout  # we may cancel dropout while testing

        self._init_weight()

    def _init_weight(self):
        """Initialize the fc's parameters"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        if self.is_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.is_dropout:
            x = self.dropout(x)
        return x

class Block(nn.Module):
    """Connect Attention and Mlp"""
    def __init__(self, config: Config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)
    
    def forward(self,x):
        """LayerNorm + MSA"""
        h = x  # for residual
        x = self.attention_norm(x)
        x = self.attn(x)
        x += h

        """LayerNorm + MLP"""
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x += h

        return x
        
    @staticmethod
    def test_input():
        blk = Block(Config())
        t = torch.randn(1, 256, 768)
        t = blk(t)
        print(f"Block test: {t.size()}")


class Test:

    @staticmethod
    def deepcopy_test():
        blk_ls =  list(Block(Config()) for _  in range(4))
        res = torch.equal(blk_ls[0].ffn.fc1.weight, blk_ls[1].ffn.fc1.weight)
        assert res == False

if __name__ == "__main__":
    Attention.test_input()
    Block.test_input()
    Test.deepcopy_test()
