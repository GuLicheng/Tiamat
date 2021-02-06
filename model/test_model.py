"""
    2020/11/27 - now
    Define your net here, create a instance and assign MODEL with it
"""
import torch.nn as nn

"""exported"""
MODEL = nn.Sequential(
    # 3*224*224  -> 3*224*224
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
    # -> 64*224*224
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    # -> 64*112*112
    nn.MaxPool2d(kernel_size=2, stride=2),
    # -> 128*112*112
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    # -> 128*224*224
    nn.UpsamplingBilinear2d(scale_factor=2),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
)


