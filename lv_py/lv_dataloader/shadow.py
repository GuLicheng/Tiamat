import torch
import numpy as np
import copy
import os
import random
import cv2 as cv
from torchvision.transforms import transforms
from skimage import measure, color, morphology
from lv_utility.image import ImageHandle

class Config:

    batch_size = 8
    num_worker = 0

    
    # root = r"F:/SaliencyMap/RGBD_for_test/DUT-RGBD/"
    # root = r"G:/DataSet/VT5000/VT5000_clear/Train/"
    root = "G:/DataSet/SBUshadow/SBUTrain/"
    test_root = r"G:/DataSet/SBUshadow/SBUTestRename/"
    dirs = (
        "ShadowImages",
        "ShadowMasks",
    )
    suffixes = (
        ".jpg",
        ".png",
        # ".jpg"
    )

    mode = "train"
    # img_size = (384, 384)
    img_size = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]  # RGB not for BGR !!!

class ProcessImageForRGB:

    def __init__(self) -> None:
        super().__init__()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=Config.mean[::-1], # opencv
                                            std=Config.std[::-1])
        


    def __call__(self, path: str, is_flip):
        image = cv.imread(path, cv.IMREAD_COLOR)
        image = cv.resize(image, Config.img_size)
        if is_flip:
            image = np.flip(image, axis=1)
            image = copy.deepcopy(image)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image
 
class ProcessImageForHSV:

    def __init__(self) -> None:
        super().__init__()
        self.to_tensor = transforms.ToTensor()
        

    def __call__(self, path: str, is_flip):
        image = cv.imread(path, cv.IMREAD_COLOR)
        image = cv.resize(image, Config.img_size)
        if is_flip:
            image = np.flip(image, axis=1)
            image = copy.deepcopy(image)
        image = self.to_tensor(image)
        return image

class ProcessImageForGroundTruth:
    def __init__(self) -> None:
        self.to_tensor = transforms.ToTensor()

    def __call__(self, path: str, is_flip):
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, dsize=Config.img_size)
        if is_flip:
            image = np.flip(image, axis=1)
            image = copy.deepcopy(image)
        image = self.to_tensor(image)
        return image

class GenerateScore:
    def __init__(self) -> None:
        pass

    def __call__(self, path: str, is_flip):
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        image, percentage = self.cal_subitizing(image)
        image = torch.Tensor([image])
        return image

    def cal_subitizing(self, label, threshold=8, min_size_per=0.005):
        # label = np.array(label.convert('1'))
        #cal number of connect areas
        dst = morphology.remove_small_objects(label, min_size=min_size_per*label.shape[0]*label.shape[1], connectivity=2)#remove small connected areas with a threshold
        labels = measure.label(dst, connectivity=2, background=0)
        number = labels.max()+1
        number = min(number, threshold)
        # number_per = number/threshold # for regression
        number_per = number # for classification
        #cal percentage of shadow areas
        percentage = np.sum(label)/(label.shape[0]*label.shape[1])
        return number_per, percentage

class DataSet:

    def __init__(self, 
        image = r"G:\DataSet\SBUshadow\SBUTrain\ShadowImages", 
        label = r"G:\DataSet\SBUshadow\SBUTrain\ShadowMasks",
        train = True) -> None:

        self.train = train

        self.image = image
        self.label = label
        

        # self.image = r"G:\DataSet\Userdefined\RGBS"
        # self.label = r"G:\DataSet\Userdefined\RGBLabel"


        self.image = [os.path.join(self.image, file) for file in os.listdir(self.image)]
        self.label = [os.path.join(self.label, file) for file in os.listdir(self.label)]

        

        self.transRGB = ProcessImageForRGB()
        self.transGT = ProcessImageForGroundTruth()
        self.score_generator = GenerateScore()


        if train:
            assert self.image.__len__() == self.label.__len__()
        print("Succseefully read ", self.image.__len__(), " samples")



    def __len__(self):
        return self.image.__len__()

    def __getitem__(self, idx):
        img = self.image[idx]
        gt = self.label[idx]

        score = self.score_generator(gt, False) # Dummy second params

        is_flip = random.random() > 0.5 
        assert is_flip == True or is_flip == False

        if self.train == False:  
            is_flip = False

        img = self.transRGB(img, is_flip)
        gt = self.transGT(gt, is_flip)

        return img, gt, score

    def file_iter(self):
        
        class FileIterator:

            def __init__(self, data_set: DataSet) -> None:
                self.data_set = data_set

            def __iter__(self):
                self.cur = 0
                return self

            def __next__(self):
                if self.cur >= self.data_set.__len__():
                    raise StopIteration()
                else:
                    filepath = self.data_set.image[self.cur]
                    img, gt, _ = self.data_set[self.cur]
                    self.cur += 1
                    return img, gt, filepath

        return FileIterator(self)


class DataSet2:

    def __init__(self, 
        image = r"G:\DataSet\SBUshadow\SBUTrain\ShadowImages", 
        label = r"G:\DataSet\SBUshadow\SBUTrain\ShadowMasks",
        mask = r"G:\DataSet\SBUshadow\SBUTrain\ShadowHSV",
        train = True) -> None:

        self.train = train

        self.image = image
        self.label = label
        self.mask = mask


        self.image = ImageHandle.files(self.image)
        self.label = ImageHandle.files(self.label)
        self.mask = ImageHandle.files(self.mask)
        

        self.transRGB = ProcessImageForRGB()
        self.transGT = ProcessImageForGroundTruth()
        self.transMask = ProcessImageForHSV()


        if train:
            assert self.image.__len__() == self.label.__len__() == self.mask.__len__()
        print("Succseefully read ", self.image.__len__(), " samples")


    def __len__(self):
        return self.image.__len__()

    def __getitem__(self, idx):
        img = self.image[idx]
        gt = self.label[idx]
        mask = self.mask[idx]

        is_flip = random.random() > 0.5 
        # assert is_flip == True or is_flip == False

        if self.train == False:  
            is_flip = False

        img = self.transRGB(img, is_flip)
        gt = self.transGT(gt, is_flip)
        mask = self.transMask(mask, is_flip)

        return img, gt, mask

    def file_iter(self):
        
        class FileIterator:

            def __init__(self, data_set: DataSet) -> None:
                self.data_set = data_set

            def __iter__(self):
                self.cur = 0
                return self

            def __next__(self):
                if self.cur >= self.data_set.__len__():
                    raise StopIteration()
                else:
                    filepath = self.data_set.image[self.cur]
                    img, gt, mask = self.data_set[self.cur]
                    self.cur += 1
                    return img, gt, mask, filepath

        return FileIterator(self)