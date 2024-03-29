from typing import List, Tuple
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


#################################################################
#                 Version 1.0
#################################################################
"""
    SomeKeys:
        image 
        semantic -> semantic annonation 
        scribble -> scribble annonation
        class -> classification
        saliency -> saliency map
"""

KEYS = [
    "image", "semantic", "scribble", "class", "saliency"
]


RESIZE_MODE = {
    "image": InterpolationMode.BILINEAR,
    "semantic": InterpolationMode.NEAREST,
    "scribble": InterpolationMode.NEAREST,
    "saliency": InterpolationMode.NEAREST,
}

class OperationFilter:

    def __init__(self, args: List[str]) -> None:
        self.args = args
        
        # check keys
        for key in args: assert key in KEYS, f"Expected one of {KEYS}, but got {key}"
        assert len(self.args) > 0

class Resize(OperationFilter):

    def __init__(self, args: List[str], size: Tuple[int, int]) -> None:
        super().__init__(args)

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, Tuple):
            self.size = size
        else:
            assert False, f"Expected int or Tuple but got {type(size)}"


    def __call__(self, sample):
        for name in self.args:
            # sample[name] = sample[name].resize(self.size, RESIZE_MODE[name])
            sample[name] = F.resize(sample[name], self.size, RESIZE_MODE[name])

        return sample

class RandomHorizontalFlip(OperationFilter):

    def __init__(self, args, ratio = 0.5) -> None:
        super().__init__(args)
        self.ratio = ratio
        assert 0 < self.ratio < 1, f"ratio should between (0, 1)"

    def __call__(self, sample):

        if random.random() < self.ratio:
            for k in self.args:
                sample[k] = sample[k].transpose(Image.FLIP_LEFT_RIGHT)

        return sample

class RandomScaleCrop(OperationFilter):

    def __init__(self, args, size: Tuple[int, int], scale=(0.5, 2.0), ratio=(3. / 4., 4. / 3.)):
        super().__init__(args)
        self.size = size
        self.scale = scale
        self.ratio = ratio


    def __call__(self, sample):
        
        i, j, h, w = transforms.RandomResizedCrop.get_params(sample["image"], self.scale, self.ratio)

        for name in self.args:
            sample[name] = F.resized_crop(sample[name], i, j, h, w, self.size, RESIZE_MODE[name])

        return sample

class ToTensor(OperationFilter):

    def __init__(self, args) -> None:
        super().__init__(args)

    def __call__(self, sample):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # => transpose(2, 0, 1)

        for name in self.args:
            if name in ["image"]:
                sample[name] = torch.from_numpy(np.array(sample[name]).astype(np.float32).transpose((2, 0, 1))).float()  
            else:
                sample[name] = torch.from_numpy(np.array(sample[name]).astype(np.float32)).float()  

        return sample

class NormalizeImage(OperationFilter):
    # (standardization)
    # MEAN = (0.485, 0.456, 0.406)
    # STD = (0.229, 0.224, 0.225)

    # https://mmsegmentation.readthedocs.io/en/latest/tutorials/config.html
    MEAN = [123.675, 116.28, 103.53]
    STD = [58.395, 57.12, 57.375]

    # image after ToTensor will not scaled to [0, 1] with type np.uint8

    def __init__(self, args = ["image"], mean = MEAN, std = STD) -> None:
        super().__init__(args)
        self.mean = mean  
        self.std = std
        self.normalizer = transforms.Normalize(mean=self.mean, std=self.std)

    def __call__(self, sample):

        for name in self.args:
            sample[name] = self.normalizer(sample[name])

        return sample

class ReadImage(OperationFilter):

    def __init__(self, args = ["image"]) -> None:
        super().__init__(args)

    def __call__(self, sample: dict):

        for arg in self.args:

            image_path = sample[arg]
            sample[f"{arg}_path"] = image_path 
            sample[arg] = Image.open(image_path).convert("RGB")

        return sample

class ReadAnnotation(OperationFilter):

    def __init__(self, args = ["semantic"]) -> None:
        super().__init__(args)

    def __call__(self, sample):
                
        for arg in self.args:
            anno_path = sample[arg]
            sample[f"{arg}_path"] = anno_path 
            sample[arg] = Image.open(anno_path)
        return sample

class NormalizeSaliencyMap(OperationFilter):

    def __init__(self, args = ["saliency"]) -> None:
        super().__init__(args)

    def __call__(self, sample):
        
        sample["saliency"] /= 255.0

        return sample

class CenterCrop(OperationFilter):

    def __init__(self, args: List[str], size: Tuple[int, int]) -> None:
        super().__init__(args)
        self.size = size

    def __call__(self, sample):

        for name in self.args:
            sample[name] = F.center_crop(sample[name], self.size)       

        return sample

class ColorJitterImage(OperationFilter):

    def __init__(self, brightness, contrast, saturation, hue, args=["image"]) -> None:
        super().__init__(args)

        self.fn = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, sample):

        for name in self.args:
            sample[name] = self.fn(sample[name])

        return sample





