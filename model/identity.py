import torch.nn as nn

class Identity(nn.Module):
    """
        A Identity map
    """
    def __init__(self, *args):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x