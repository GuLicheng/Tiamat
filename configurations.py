"""
    2020/12/01 - 2021/2/5

    This file provide the parameters for initializing your model

    This constant in this file was deprecated, please change your configs in 
    Configration/config.ini

"""

"""

# the size of image
RESIZE = (224, 224)

# batch_size of data_loader
BATCH_SIZE = 32

# num_work of data_loader
NUM_WORKERS = 1



    class torchvision.transform.ToTensor
    PIL.Image or numpy.nd-array data whose pixel value range of shape=(H,W,C) is [0, 255]
    converted into pixel data of shape=(C,H,W) and normalized to the torch.FloatTensor type of [0.0, 1.0].

    ImageNet:
        channel =     R      G      B     
        mean    =  [0.485, 0.456, 0.406]
        std     =  [0.229, 0.224, 0.225]
    the average and standard variance for RGB channels pretrained from ImageNet

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


"""

"""Here is formal code:"""

import configparser
import torch
import torch.optim
from dataclasses import dataclass
from typing import *

@dataclass(init=True, unsafe_hash=False, repr=False, eq=False, order=False, frozen=False)
class Config:

    def __init__(self, path) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        """Read Configurations"""
        config = configparser.ConfigParser()
        config.read(path)

        # training set path
        self.secondary_directory_train_root: str = config.get("path", "train_root")
        self.secondary_directory_train_paths: Tuple[str, ...] = (
            config.get("path", "train_sample_path"),
            config.get("path", "train_label_path")
        )
        self.secondary_directory_train_suffixes: Tuple[str, ...] = (
            config.get("path", "train_sample_suffix"),
            config.get("path", "train_label_suffix")
        )

        # testing set path
        self.secondary_directory_test_root: str = config.get("path", "test_root")
        self.secondary_directory_test_paths: Tuple[str, ...] = (
            config.get("path", "test_sample_path"),
            config.get("path", "test_label_path")
        )
        self.secondary_directory_test_suffixes: Tuple[str, ...] = (
            config.get("path", "test_sample_suffix"),
            config.get("path", "test_label_suffix")
        )


        self.size = (
            config.getint("size", "weight"), 
            config.getint("size", "height")
        )
        self.mean = (
            config.getfloat("mean", "red"),
            config.getfloat("mean", "green"),
            config.getfloat("mean", "blue")
        )   
        self.std = (
            config.getfloat("std", "red"),
            config.getfloat("std", "green"),
            config.getfloat("std", "blue")
        )
        self.batch_size = config.getint("basic", "batch_size")
        self.num_workers = config.getint("basic", "num_worker")
        self.learning_rate = config.getfloat("basic", "learning_rate")
        self.dropout = config.getfloat("basic", "dropout")

"""exported"""
config = Config(r"D:\GraphTheoryCode\Template\Tiamat\configuration\config.ini")

if __name__ == "__main__":
    print(config.batch_size)
    print(config.num_workers)
    print(config.learning_rate)
    print(config.dropout)
    print(config.mean)
    print(config.std)
    print(config.size)
    print(config.device)
    print(config.secondary_directory_train_paths)
    print(config.secondary_directory_train_suffixes)