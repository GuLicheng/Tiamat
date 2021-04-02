# c o d i n g = utf-8
import os
import torch.nn as nn
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import random
import dataclasses

@dataclasses.dataclass(init=True, unsafe_hash=False, repr=True, eq=False, order=False, frozen=False)
class Config:
    def __init__(self) -> None:
        """
            For ("D:/MY/DataSet/CoSal2015/Image/class1/0001.jpg", "D:/MY/DataSet/CoSal2015/GroundTruth/class1/0001.png")
            data_root = "D:/MY/DataSet/CoSal2015/"
            image_file_name = "Image"
            ground_truth_name = "GroundTruth"

        The directory structure such as:
        data_root
            ├── Image
            │     ├── class1
            │     │     ├── 00001.jpg
            │     │     ├── 00002.jpg
            │     │     └── 00003.jpg
            │     │
            │     └── class2
            │             ├── 00001.jpg
            │             ├── 00002.jpg
            │             └── 00003.jpg
            └── GroundTruth
                    ├── class1
                    │     ├── 00001.jpg
                    │     ├── 00002.jpg
                    │     └── 00003.jpg
                    │
                    └── class2
                            ├── 00001.jpg
                            ├── 00002.jpg
                            └── 00003.jpg    

        """

        self.data_root = r"D:/MY/DataSet/CoSal2015/"
        # self.data_root = r"D:/MY/DataSet/iCoseg"
        self.image_file_name = r"Image"
        self.ground_truth_name = r"GroundTruth"

        self.image_file_suffix = r".jpg"
        self.ground_truth_suffix = r".png"

        self.img_size = (256, 256)
        self.batch_size = 1
        self.num_thread = 0
        self.sample_num = 5

config = Config()

class Debugger(nn.Module):
    def __init__(self):
        super(Debugger, self).__init__()

    def forward(self, x):
        """
            Some operations for track x
        """    
        return x


class CoData(data.Dataset):
    def __init__(self):
        self.img_root = os.path.join(config.data_root, config.image_file_name)
        self.label_root = os.path.join(config.data_root, config.ground_truth_name)
        self.class_list = os.listdir(self.img_root)
        self.size = config.img_size
        self.img_dir = list(
            map(lambda x: os.path.join(self.img_root,x), self.class_list))   #ͼƬĿ¼��ַ
        self.label_dir = list(
            map(lambda x: os.path.join(self.label_root, x), self.class_list))  # ��ǩĿ¼��ַ
        self.img_name_list = [os.listdir(idir) for idir in self.img_dir]
        self.gt_name_list = [os.listdir(idir) for idir in self.label_dir]
        self.img_path_list = [
            list(
                map(lambda x: os.path.join(self.img_dir[idx], x),
                    self.img_name_list[idx]))
            for idx in range(len(self.img_dir))
        ]
        self.gt_path_list = [
            list(
                map(lambda x: os.path.join(self.label_dir[idx], x),
                    self.gt_name_list[idx]))
            for idx in range(len(self.label_dir))
        ]
        self.img_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.label_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        img_paths = self.img_path_list[item]
        label_paths = self.gt_path_list[item]
        num = len(img_paths)
        """
            if the size of img_paths greater than sample_num, sample sample_num, 
            otherwise take all samples
        """
        if num > config.sample_num:
            sampled_list = random.sample(range(num), config.sample_num)
            new_img_paths = [img_paths[i] for i in sampled_list]
            img_paths = new_img_paths
            new_gt_paths = [label_paths[i] for i in sampled_list]
            label_paths = new_gt_paths
            num = config.sample_num
        else:
            assert False, "Each group must greater than sample_num"

        imgs = torch.Tensor(num, 3, self.size[0], self.size[1])
        labels = torch.Tensor(num, 1, self.size[0], self.size[1])

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            label = Image.open(label_paths[idx]).convert('L')
            img = self.img_transform(img)
            imgs[idx] = img
            label = self.label_transform(label)
            labels[idx] = label

        return imgs, labels

    def __len__(self):
        return len(self.class_list)

def get_loader(pin=False):
    shuffle = True
    dataset = CoData()
    data_loader = data.DataLoader(dataset=dataset, batch_size = config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)

    return data_loader

"""exported loader"""
TRAINING_LOADER = get_loader()

if __name__ == "__main__":
    path = r"D:/MY/DataSet/CoSal2015/"
    loader = get_loader()
    for batch, (x, y) in enumerate(loader):
        print(x.size(), y.size())
        break

