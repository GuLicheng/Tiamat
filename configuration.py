"""
    2020/11/27 - now
    all configuration for your model here
"""
import torch.nn as nn
import torch.optim

from loader.data_loader import TRAIN_LOADER, TEST_LOADER

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

OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=1e-3)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    "train_loader": TRAIN_LOADER,
    "test_loader": TEST_LOADER,
    "device": DEVICE,
    "optimizer": OPTIMIZER,
    "model": MODEL
}

if __name__ == '__main__':
    print(cfg.keys())
    print(cfg["device"])