import configparser
import torch
from dataclasses import dataclass
from typing import *


@dataclass(init=True, unsafe_hash=False, repr=True, eq=False, order=False, frozen=False)
class Config:

    def __init__(self, path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """Read Configurations"""
        config = configparser.ConfigParser()
        config.read(path)

        # training set path

        self.image_path = config.get("path", "trainset_image_path")
        self.gt_path = config.get("path", "trainset_gt_path")
        
        self.group_sample = 20

        self.size = (256, 256)
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
        self.batch_size = 1
        self.num_workers = 4
        self.learning_rate = 0.001
        self.dropout = 0.1
        self.epoch = 150


"""exported"""
config = Config(r"D:\MY\SOD\Tiamat\configuration\co_saliency.ini")

if __name__ == "__main__":
    for item in config.__dict__.items():
        print(item)
