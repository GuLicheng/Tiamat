from tqdm import tqdm
from utils.metric import Evaluator
from dataset.pascalvoc import PascalVoc, make_pascal_voc
import numpy as np
from PIL import Image
import os
import cv2 as cv


if __name__ == "__main__":

    ds = make_pascal_voc(data_root=r"D:\dataset\VOCdevkit\VOC2012", split="scribble_val.txt")


    print(ds.__dict__)
    




