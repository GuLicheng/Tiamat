"""
    2020/11/30 - now
    This module provide the transformation(preprocess operations) 
    for your image
"""

import sys
sys.path.append("./../")

from PIL import Image
from typing import *
from torchvision import transforms
from configurations import config


# read RGB pictures
class ImageOpenConvertRGB:
    def __call__(self, x):
        return Image.open(x).convert("RGB")


# read saliency map
class ImageOpenConvert1:
    def __call__(self, x):
        return Image.open(x).convert("1")


"""
    class torchvision.transform.ToTensor
    PIL.Image or numpy.nd-array data whose pixel value range of shape=(H,W,C) is [0, 255]
    converted into pixel data of shape=(C,H,W) and normalized to the torch.FloatTensor type of [0.0, 1.0].

    ImageNet:
        channel =     R      G      B     
        mean    =  [0.485, 0.456, 0.406]
        std     =  [0.229, 0.224, 0.225]
    the average and standard variance for RGB channels pretrained from ImageNet
"""

# pre-process for image
transforms_for_image: transforms.Compose = transforms.Compose([
    # lambda x: Image.open(x).convert("RGB"),
    ImageOpenConvertRGB(),
    transforms.Resize(config.size),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.mean, std=config.std)
])

# pre-process for label
transforms_for_label: transforms.Compose = transforms.Compose([
    # lambda x: Image.open(x).convert("1"),
    ImageOpenConvert1(),
    transforms.Resize(config.size),
    transforms.ToTensor()
])
"""
    Why I don't use lambda:
    Stack Overflow:
        No, it is not supported on Windows. The reason is that multiprocessing lib 
        doesn’t have it implemented on Windows. There are some alternatives 
        like dill that can pickle more objects.

    There are some reasons that I agree with:
        So I don't use lambda, but you can use lambda inside the __getitem__. 
        I think if do that, it will create new objects 
        when __getitem__ called, and it will waste resources.
"""

"""the transforms exported finally"""
TRAIN_TRANSFORMS: Tuple[transforms.Compose, ...] = (
    transforms_for_image,
    transforms_for_label,
)

TEST_TRANSFORMS: Tuple[transforms.Compose, ...] = (
    transforms_for_image,
    transforms_for_label,
)
