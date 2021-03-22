"""
    For a three-level directory structure, we can use ImageFolder module
    which in torchvision.datasets.folder

    root
      ├── train
      │     ├── class1
      │     │     ├── 00001.jpg
      │     │     ├── 00002.jpg
      │     │     └── 00003.jpg
      │     │
      │     └── class2
      │             ├── 00001.jpg
      │             ├── 00002.jpg
      │             └── 00003.jpg
      └── test
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


"""
  A Simply data_loader copied from somewhere, I may change it someday...
"""

import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CoData(data.Dataset):
    def __init__(self, img_root, img_size=224):

        class_list = os.listdir(img_root)
        self.size = [img_size, img_size]

        self.img_dirs = list(
            map(lambda x: os.path.join(img_root, x), class_list))


        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):

        names = os.listdir(self.img_dirs[item])
        num = len(names)
        img_paths = list(
            map(lambda x: os.path.join(self.img_dirs[item], x), names))     #??????
        # print(names)
        imgs = torch.Tensor(num, 3, self.size[0], self.size[1])    #4???????

        subpaths = []
        ori_sizes = []

        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            subpaths.append(
                os.path.join(img_paths[idx].split('/')[-2],
                             img_paths[idx].split('/')[-1][:-4] + '.png'))    #?????+?????????.jpg??+.png
            ori_sizes.append((img.size[1], img.size[0]))
            img = self.transform(img)
            imgs[idx] = img    #?›¥????????????

        return imgs, subpaths, ori_sizes

    def __len__(self):
        return len(self.img_dirs)


def get_loader(img_root, img_size, num_workers=4, pin=True):
    dataset = CoData(img_root, img_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader

# G:\COCO-SEG\train\groundtruth
# G:\COCO-SEG\train\image
trainset_image_path = r"G:/COCO-SEG/train/image"
trainset_gt_path = r"G:/COCO-SEG/train/groundtruth"


"""exported loader"""
CO_SALIENCY_TRAIN_IMAGE_LOADER = DataLoader(CoData(trainset_image_path, 256), batch_size=1,num_workers=4)
CO_SALIENCY_TRAIN_GT_LOADER = DataLoader(CoData(trainset_gt_path, 256), batch_size=1,num_workers=4)



if __name__ == "__main__":
    cnt = 3
    for img, gt in zip(CO_SALIENCY_TRAIN_IMAGE_LOADER, CO_SALIENCY_TRAIN_GT_LOADER):
        assert img[0].size()[1] == gt[0].size()[1]
        print(img[0].size(), gt[0].size())
        if cnt == 0:
            break
        cnt -= 1


