import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.nn.modules.normalization import LayerNorm
# import seaborn
# seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module):
    """
        A standard Encoder-Decoder architecture. Base for this and many
        other models
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Talk in and process masked src and target sequences"""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    """Define standard linear + softmax generation step"""
    def __init__(self, d_model, vocal):
        # d_model -> inchannels, vocal -> outchannels
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocal)

    def forward(self, x):
        # softmax for last dimension
        return F.log_softmax(self.proj(x), dim=-1)

def clone(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module)] for _ in range(N))

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N=6):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class LayerNorm(nn.Module):
    """Constructure a layernorm module (See citation for detail)"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """a * (x - mean) / std + b"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SubLayerConnection(nn.Module):
    """
        A residual connection followed by a layer norm
        Note for code simplicity the norm is first as opposed to last
    """
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = dropout

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with same size"""
        return x + self.dropout(sublayer(self.norm(x)))

"""
　 　每层都有两个子层组成。
    第一个子层实现了“多头”的 Self-attention，
    第二个子层则是一个简单的Position-wise的全连接前馈网络。　　
"""

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attention and feed forward(defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SubLayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections"""
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    """Generic N layer decoder with masking"""
    def __init__(self, layer, N=6):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn and feed forward(defined below)"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections"""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    """Comute "Scaled Dot Product Attention" """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask==0, -)