import torch
import torch.nn as nn

class SaliencyMapGenerator(nn.Module):
    """
        generator saliency map via changing the channels from input to 1, 
        and predict maps with sigmoid function
    """
    def __init__(self, channels: int):
        super(SaliencyMapGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)
    