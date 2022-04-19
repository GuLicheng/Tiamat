from tqdm import tqdm
from lv_py.utils.metric import Evaluator
import numpy as np
from PIL import Image
import os
import cv2 as cv

if __name__ == "__main__":


    
    saliency = r"D:\experiment\saliency\saliency_mask"
    semantic = r"D:\dataset\VOCdevkit\VOC2012\SegmentationClass"
    image = r"D:\dataset\VOCdevkit\VOC2012\JPEGImages"
    split = r"D:\dataset\VOCdevkit\VOC2012\scribble_train.txt"

    masks = np.load("refined_attention_masks.npy", allow_pickle=True).item()

    for name in tqdm(open(split, "r").read().splitlines()): 

        gt = np.array(Image.open(f"{os.path.join(semantic, name)}.png"))
        gt[gt != 0] = 255
        sal = cv.imread(f"{os.path.join(saliency, name)}.png")

        sal[sal != 0] = 1

        mask = masks[name]


        





