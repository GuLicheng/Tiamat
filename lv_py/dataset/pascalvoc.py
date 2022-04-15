from typing import Iterable
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import warnings
import os
from tqdm import tqdm

class PascalVoc(Dataset):
    """
        PascalVoc dataset
    """
    NUM_CLASSES = 21


    PALETTE = [
        (0, 0, 0),  # 0=background
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ]

    CAT_LIST_SCRIBBLE = [
        'background', 'plane', 'bike', 'bird', 'boat', 
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog', 
        'horse', 'motorbike', 'person', 'plant', 'sheep', 
        'sofa', 'train', 'monitor'
    ]

    CAT_LIST = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor'
    ]

    def __init__(self, img_dir, split, anno_dir = None, class_label = None, sal_dir = None, pipeline = None):

        super().__init__()

        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.sal_dir = sal_dir
        self.splits = open(split, "r").read().splitlines()
        self.pipeline = pipeline
        self.class_label = class_label

        self.images, self.semantics, self.class_labels, self.saliency = [None] * 4

        self._collect_images()._collect_semantic()._collect_class_label()._collect_saliency()

        print(f"sample num: {len(self.splits)}")


    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        
        if isinstance(index, slice): # debug
            return self._slice(index)
        else:
            return self._getitem(index)

    def reset(self, splits: Iterable[str]):
        self.splits = list(splits)
        self.images, self.semantics, self.class_labels, self.saliency = [None] * 4
        self._collect_images()._collect_semantic()._collect_class_label()._collect_saliency()

    def _slice(self, index):
        """
            Simplely modify self.splits
        """
        self.splits = self.splits[index]
        self.images = self.images[index]
        if self.semantics is not None:
            self.semantics = self.semantics[index]
        if self.sal_dir is not None:
            self.saliency = self.saliency[index]
        warnings.warn("this dataset has been modified!")
        return self

    def _getitem(self, index):
        sample = {
            "image": self.images[index],
            "ori_name": self.splits[index],
        }

        if self.semantics is not None:
            sample["semantic"] = self.semantics[index]

        if self.class_labels is not None:
            sample["class"] = self.class_labels[self.splits[index]]

        if self.sal_dir is not None:
            sample["saliency"] = self.saliency[index]

        return self.pipeline(sample)

    def _collect_images(self):
        self.images = [f"{self.img_dir}/{name}.jpg" for name in self.splits]
        return self

    def _collect_semantic(self):
        if self.anno_dir is not None:
            self.semantics = [f"{self.anno_dir}/{name}.png" for name in self.splits]
        return self

    def _collect_class_label(self):
        if self.class_label is not None:
            self.class_labels = np.load(self.class_label, allow_pickle=True).item()
        return self

    def _collect_saliency(self):
        if self.sal_dir is not None:
            self.saliency = [f"{self.sal_dir}/{name}.png" for name in self.splits]
        return self

    @staticmethod
    def decode_segmap(image: np.ndarray, use_rgb = True):

        assert isinstance(image, np.ndarray)
        
        label_colors = np.array(PascalVoc.PALETTE)

        r = np.full_like(image, fill_value=255).astype(np.uint8)
        g = np.full_like(image, fill_value=255).astype(np.uint8)
        b = np.full_like(image, fill_value=255).astype(np.uint8)

        for l in range(0, PascalVoc.NUM_CLASSES):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        
        rgb = np.stack([r, g, b], axis=2)
        if use_rgb:
            rgb = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        return rgb

    @staticmethod
    def show_image(image: torch.Tensor) -> np.ndarray:

        """
            # (standardization)
            # MEAN = (0.485, 0.456, 0.406)
            # STD = (0.229, 0.224, 0.225)

            # https://mmsegmentation.readthedocs.io/en/latest/tutorials/config.html
            MEAN = [123.675, 116.28, 103.53] 255 times larger than above
            STD = [58.395, 57.12, 57.375]
        """

        def denormalizeimage(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            """Denormalize tensor images with mean and standard deviation.
            Args:
                images (tensor): N*C*H*W
                mean (tuple): means for each channel.
                std (tuple): standard deviations for each channel.
            """
            images = images.cpu().numpy()
            # N*C*H*W to N*H*W*C
            images = images.transpose((0, 2, 3, 1))
            images *= std
            images += mean
            images *= 255.0
            # N*H*W*C to N*C*H*W
            images = images.transpose((0, 3, 1, 2))
            return torch.tensor(images)

        assert isinstance(image, torch.Tensor)

        if len(image.shape) == 3:
            image = image[None, :, :, :]

        image = denormalizeimage(image)
        assert image.shape[0] == 1

        rgb = image[0].permute(1, 2, 0).contiguous().detach().cpu().numpy().astype(np.uint8)

        return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)


    @staticmethod
    def show_semantic(image: torch.Tensor) -> np.ndarray:

        assert isinstance(image, torch.Tensor) and len(image.shape) < 4
        assert ((PascalVoc.NUM_CLASSES <= image) & (image < 255)).sum() == 0
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image[0]

        rgb = image.detach().cpu().numpy()
        rgb = PascalVoc.decode_segmap(rgb, use_rgb=True)

        return rgb

    @staticmethod
    def make_class_label_from_xmls(xml_dir: str, dest: str):
        
        CAT_NAME_TO_NUM = dict(map(lambda kv: (kv[1], kv[0]), enumerate(PascalVoc.CAT_LIST)))
        
        cls_label = {}

        xmls = os.listdir(xml_dir)

        for xml in tqdm(xmls):
            ele_tree = ET.parse(os.path.join(xml_dir, xml))
            name = ele_tree.findtext("filename")[:-4]   # remove suffix `.jpg`

            label = [0.0] * (PascalVoc.NUM_CLASSES - 1)
            for cls in ele_tree.findall("object"):
                label[CAT_NAME_TO_NUM[cls.findtext("name")]] = 1.0
            cls_label[name] = np.array(label)

        np.save(f"{dest}/cls_labels.npy", cls_label)
        return cls_label

    @staticmethod
    def make_scribble_from_xmls(xml_dir: str, dest: str, thickness = 3):

        CAT_NAME_TO_NUM = dict(map(lambda kv: (kv[1], kv[0]), enumerate(PascalVoc.CAT_LIST_SCRIBBLE)))

        def make_scribble_from_xml(xml: str, thickness: int = 3, unlabeled: int = 255):

            def pairwise(iterable):
                import itertools
                a, b = itertools.tee(iterable)
                next(b, None)
                return zip(a, b)

            def clamp(x, lower, upper):
                return min(max(x, lower), upper)

            xml = ET.parse(xml)

            size_info = xml.find("size")
            width = int(size_info.findtext("width"))
            height = int(size_info.findtext("height"))

            mask = np.full(shape=(height, width), dtype=np.uint8, fill_value=unlabeled)

            for polygon in xml.findall("polygon"):
                tag = polygon.findtext("tag")
                for point1, point2 in pairwise(polygon.findall("point")):
                    x1, y1 = int(point1.findtext("X")), int(point1.findtext("Y"))
                    x2, y2 = int(point2.findtext("X")), int(point2.findtext("Y"))

                    x1, x2 = clamp(x1, 0, width), clamp(x2, 0, width)
                    y1, y2 = clamp(y1, 0, height), clamp(y2, 0, height)

                    cv.line(mask, (x1, y1), (x2, y2), CAT_NAME_TO_NUM[tag], thickness=thickness)


            name = xml.findtext("filename")[:-4]    # remove suffix `.jpg`
            return name, mask

        for xml in tqdm(os.listdir(xml_dir)):
            name, scribble = make_scribble_from_xml(os.path.join(xml_dir, xml), thickness)
            cv.imwrite(f"{os.path.join(dest, name)}.png", scribble)

    @staticmethod
    def make_PIL(image: np.ndarray, dest: str, name: str):

        import itertools
        palette = itertools.chain(
            itertools.chain.from_iterable(PascalVoc.PALETTE),
            itertools.repeat(255, 3 * (256 - PascalVoc.NUM_CLASSES)),  # fill (255, 255, 255) for unknown class
        )

        img = Image.fromarray(np.uint8(image))
        img.putpalette(palette)
        img.save(f"{dest}/{name}.png")



    from dataset.pipeline import (ReadImage, RandomScaleCrop, 
    RandomHorizontalFlip, ColorJitterImage, ToTensor, NormalizeImage, ReadAnnotation)
    from torchvision.transforms import transforms

    """Here are some pipelines"""
    IMAGE_LEVEL_TRAIN = transforms.Compose([
        ReadImage(),
        RandomScaleCrop(args=["image"], scale=(0.5, 1.0), size=((512, 512))),
        RandomHorizontalFlip(args=["image"]),
        ColorJitterImage(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ToTensor(args=["image"]),
        NormalizeImage(args=["image"]),
    ])

    IMAGE_LEVEL_VAL = transforms.Compose([
        ReadImage(),
        ToTensor(args=["image"]),
        NormalizeImage(),
    ])

    SEMANTIC_TRAIN = transforms.Compose([
        ReadImage(),
        ReadAnnotation(),
        RandomHorizontalFlip(args=["image", "semantic"]),
        RandomScaleCrop(args=["image", "semantic"], size=((512, 512))),
        ToTensor(args=["image", "semantic"]),
        NormalizeImage(),
    ])

    SEMANTIC_VAL = transforms.Compose([
        ReadImage(),
        ReadAnnotation(),
        ToTensor(args=["image", "semantic"]),
        NormalizeImage(),
    ])


if __name__ == "__main__":

    import sys 
    sys.path.append("/home/zeyu_yan/lv_py") 
    from config.dataset_cfg import MY77

    parser = MY77()
    args = parser.parse_args()

    dataset = PascalVoc(
        img_dir=args.image_directory,
        split=args.val_txt,
        class_label=args.image_level_npy,
        anno_dir=args.semantic_directory,
        pipeline=PascalVoc.SEMANTIC_VAL
    )

    sample = dataset[1]

    image = sample["image"]
    ori_name = sample["ori_name"]
    semantic = sample["semantic"]


    semantic = semantic.numpy()
    PascalVoc.make_PIL(semantic, "./", "simple_test")

    print("Ok for `pascal_voc_test`")





















