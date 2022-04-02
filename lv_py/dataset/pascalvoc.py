import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
import torch
import warnings

class PascalVoc(Dataset):
    """
        PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self, img_dir, split, anno_dir = None, class_label = None, pipeline = None):

        super().__init__()

        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.splits = open(split, "r").read().splitlines()
        self.pipeline = pipeline
        self.class_label = class_label

        self.images, self.semantics, self.class_labels = [None] * 3

        self._collect_images()._collect_semantic()._collect_class_label()

        print(f"sample num: {len(self.splits)}")


    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        
        if isinstance(index, slice): # for debug
            return self._slice(index)
        else:
            return self._getitem(index)


    def _slice(self, index):
        """
            Simplely modify self.splits
        """
        self.splits = self.splits[index]
        self.images = self.images[index]
        if self.semantics is not None:
            self.semantics = self.semantics[index]
        warnings.warn("this dataset has been modified!")
        return self

    def _getitem(self, index):
        sample = {
            "image": self.images[index],
            "ori_name": self.splits[index]
        }

        if self.semantics is not None:
            sample["semantic"] = self.semantics[index]

        if self.class_labels is not None:
            sample["class"] = self.class_labels[self.splits[index]]

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

    @staticmethod
    def decode_segmap(image: np.ndarray, use_rgb = True):

        assert isinstance(image, np.ndarray)
        
        label_colors = np.array([(0, 0, 0),  # 0=background
                    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

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




























