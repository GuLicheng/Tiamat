"""
--------------Do not change anything from here downwards!------------------

    Time: 2020/11/10 - now

    This is a basic framework for user-defined DataLoader, you
    should overload the __init__, __getitem__ and __len__.

    __init__: 
        1.initialize the file path and list of file names.

    __getitem__: 
        1. Read all data from file.(using numpy.from, PIL.Image.open or other modules)
        2. Preprocess the data.(torchvision.Transform)
        3. Returns with a tuple of data(image and label, or maybe other info)

    __len__:
        1. Returns with the length of your dataset

    Notice:
        this template only adapt for a secondary directory such as:
        root
          ├── image_dir
          │     ├── 00001.jpg
          │     ├── 00002.jpg
          │     └── 00003.jpg
          ├── label_dir
          │     ├── 00001.png
          │     ├── 00002.png
          │     └── 00003.png
          │
          ├── edge_dir
          │     ├── 00001.png
          │     ├── 00002.png
          │     └── 00003.png
          ......

        but a three-level directory structure(all the file name must be the same).
"""
from os import path

import csv
import functools
import operator
import os

from torch.utils.data import Dataset, DataLoader
from typing import *
from torchvision import transforms
from PIL import Image
# from component.imagepath import *
# from component.transform import *
# from configuration.rgb_config import *


class SecondDirectoryStructureDataLoader(Dataset):
    """
        This class will load DUTS_TE from the given file path as following steps:
        1. Create a file.txt and save all paths of sample and label 
        2. each row of file.txt will be Image_Path, Label_Path

        In our terms, "sample" is same as "picture"

        e.g.
        "D:\MY\DataSet\DUTS\DUTS-TE\DUTS-TE-Image\ILSVRC2012_test_00000003.jpg"
        root(first directory) = D:\MY\DataSet\DUTS\DUTS-TE\ 
        image_dir(second directory) = DUTS-TE-Image\ 
        filename(sample name) = ILSVRC2012_test_00000003.jpg
        This is a secondary directory structure template
    """

    def __init__(self,
                 root: str,
                 dirs: Tuple[str, ...],
                 tfs: Tuple[transforms.Compose],
                 suffixes: Tuple[str, ...],
                 mode: str,
                 debug=False) -> None:
        """
            Parameters:
                mode: "train" or "test"
                root: seen notice
                dirs: seen notice
                tfs: transforms for sample(image)
                suffixes: seen notice
                debug: ignore it...
        """
        self.mode: str = mode
        self.root: str = root
        self.dirs: Tuple[str] = dirs
        self.tfs: Tuple[transforms.Compose] = tfs
        self.suffixes: Tuple[str, ...] = suffixes
        self.image_label_list: List[Tuple[str, ...]] = self.load_file()

        self.assert_()
        if debug:
            self.debug()

    def __len__(self):
        """
            Parameters: None

            Return:
                the number of samples
        """
        return self.image_label_list.__len__()

    def __getitem__(self, index):
        """
           load a sample from self.image_label_list and transform it
           Parameters: None

           Return:
               a pair of sample and a saliency map
        """
        t: Tuple = self.image_label_list[index]

        ls = [self.root + directory + '/' + image_name for (directory, image_name) in zip(self.dirs, t)]

        return tuple([tf(path) for (tf, path) in zip(self.tfs, ls)])

    def load_file(self) -> List[Tuple[str, ...]]:
        """
            check whether filename is available, if available, create a csv file(file.txt) with the
            name of each sample in "./data" directory

            Return:
                the name list of samples and saliency map
            Warning:
                filename(path) must be available, or this routine may throw an exception
        """
        if not os.path.exists(f'./data/{self.mode}file.csv'):
            print(f'not exists "{self.mode}file.txt", creating a "{self.mode}file.txt".....')
            self.write_csv()
        return self.read_csv()

    def write_csv(self) -> None:
        """
            Create a file.txt and read it into self.image_label_list
            Return: None
        """
        if os.path.exists(self.root):
            print('sample path exists')
            paths = [self.root + directory for directory in self.dirs]
            csv_path = f'./data/{self.mode}file.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=' ')

                # for sample xxxxxxx.suffix(such as 1.jpg),
                # there must be a label called xxxxxxx.suffix(such as 1.png)
                # get image(sample) name from image(sample) path
                # for 
                #   sample(a.jpg, b.jpg, c.jpg), 
                #   label(a.jpg, b.jpg),    // lack of c.jpg
                #   depth(a.jpg)            // lack of b.jpg and c.jpg
                # we just need (a) which is the result of intersection of sample, label, and depth
                sets = [set(map(lambda x: x.split(".")[0], os.listdir(paths[i]))) for i in range(len(self.dirs))]
                sets = functools.reduce(operator.and_, sets)
                # you can merge these in one row but I think it's too long and complex
                # for reader with few functional programming experience

                for name in sets:
                    writer.writerow([name + suffix for suffix in self.suffixes])
                print(f"Successfully written into csv file with mode {self.mode}")
        else:
            print(f"root path not exist, root is {self.root}")
            raise FileNotFoundError

    def read_csv(self) -> List[Tuple[str, ...]]:
        """
            Read the csv('./data/file.csv') file
            Return:
                the name list of samples
        """
        csv_path = f"./data/{self.mode}file.csv"
        ls = []

        if not os.path.exists(csv_path):
            print(f"{csv_path} is not exist ")
            raise FileNotFoundError

        with open(csv_path, "r") as f:
            print(f'Successfully open the file, mode is {self.mode}')
            reader = csv.reader(f)
            for line in reader:
                # Since line is a list, we use line[0] to get content
                t = line[0].split(' ')
                ls.append(tuple(t))
        print(f"Successfully read csv with {len(ls)} samples")

        return ls

    """Here are some function for test. Please ignore them"""

    def debug(self):
        self.assert_()
        self.test_for_read(0)
        self.test_for_read(1)

    # test whether sample can be load correctly
    def test_for_read(self, index) -> None:
        name, label = self.image_label_list[index]

        img_path = self.root + self.dirs[0] + "/" + name
        Image.open(img_path).convert('RGB').show()

        label_path = self.root + self.dirs[1] + "/" + label
        Image.open(label_path).show()

    def assert_(self):
        # print(self.tfs.__len__(), self.dirs.__len__(), self.suffixes.__len__())
        if len(self.tfs) == len(self.dirs) == len(self.suffixes):
            assert self.mode in ["train", "test"]
        else:
            print(len(self.tfs), len(self.dirs), len(self.suffixes))
            assert False, "Not satisfied len(self.tfs) == len(self.dirs) == len(self.suffixes)"

# """the test_loader exported finally"""
# TEST_LOADER = DataLoader(SecondDirectoryStructureDataLoader(
#     mode="test",
#     root=config.secondary_directory_test_root,
#     dirs=config.secondary_directory_test_paths,
#     tfs=TEST_TRANSFORMS,
#     suffixes=config.secondary_directory_test_suffixes), batch_size=config.batch_size, num_workers=config.num_workers)

# """"the train_loader exported finally"""
# TRAIN_LOADER = DataLoader(SecondDirectoryStructureDataLoader(
#     mode="train",
#     root=config.secondary_directory_train_root,
#     dirs=config.secondary_directory_train_paths,
#     tfs=TRAIN_TRANSFORMS,
#     suffixes=config.secondary_directory_train_suffixes), batch_size=config.batch_size, num_workers=config.num_workers)
