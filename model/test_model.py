import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d
import cv2 as cv
import numpy as np

class SpatialCGNL(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, use_scale=False, groups=8):
        super(SpatialCGNL, self).__init__()
        self.use_scale = use_scale
        self.groups = groups

        # conv theta, phi, g, z
        self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.phi = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.g = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.z = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=in_channels)

from PIL import Image
from torchvision import transforms
# read RGB pictures
class ImageOpenConvertRGB:
    def __call__(self, x):
        return Image.open(x).convert("RGB")

transforms_for_image: transforms.Compose = transforms.Compose([
    # lambda x: Image.open(x).convert("RGB"),
    ImageOpenConvertRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
path = r"C:\Users\Administrator\Pictures\Saved Pictures\01.jpg"
conv = nn.Conv2d(3, 1, kernel_size=1, stride=1)
img = transforms_for_image(path)
print(img)
img = conv(img)
img = img.numpy()
cv.imshow("show", img)