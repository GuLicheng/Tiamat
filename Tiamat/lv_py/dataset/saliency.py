import os
from torch.utils.data import Dataset

class SaliencyDataset(Dataset):

    def __init__(self, img_dir, sal_dir = None, pipeline = None) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.sal_dir = sal_dir
        self.pipeline = pipeline
        self.splits, self.images, self.saliency = [None] * 3

        self._getsplits()
        self._collect_images()
        self._collect_saliency()

    def _getsplits(self):
        self.splits = list(map(lambda x: x.split('.')[0], os.listdir(self.img_dir)))

    def _collect_images(self):
        self.images = list(map(lambda x: f"{os.path.join(self.img_dir, x)}.jpg", self.splits))

    def _collect_saliency(self):
        if self.sal_dir is not None:
            self.saliency = list(map(lambda x: f"{os.path.join(self.sal_dir, x)}.png", self.splits))

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):

        sample = {
            "ori_name": self.splits[index],
            "image": self.images[index],
            "saliency": self.saliency[index],
        }

        return self.pipeline(sample)

    from dataset.pipeline import (ReadImage, RandomScaleCrop, 
    RandomHorizontalFlip, ColorJitterImage, ToTensor, NormalizeImage, ReadAnnotation, NormalizeSaliencyMap)
    from torchvision.transforms import transforms

    SALIENCY_TRAIN = transforms.Compose([
        ReadImage(),
        ReadAnnotation(args=["saliency"]),
        RandomHorizontalFlip(args=["image", "saliency"]),
        RandomScaleCrop(args=["image", "saliency"], size=((384, 384))),
        ToTensor(args=["image", "saliency"]),
        NormalizeImage(),
        NormalizeSaliencyMap(),
    ])

    SALIENCY_VAL = transforms.Compose([
        ReadImage(),
        ReadAnnotation(args=["saliency"]),
        ToTensor(args=["image", "saliency"]),
        NormalizeImage(),
        NormalizeSaliencyMap(),
    ])


class DUTS(SaliencyDataset):
    """
        DUTS-TR
            Imgs
            GT
    """
    def __init__(self, img_root, pipeline = None) -> None:
        super().__init__(os.path.join(img_root, "Imgs"), os.path.join(img_root, "GT"), pipeline)



class SBU(SaliencyDataset):
    """
        SBUShadow
            ShadowImages
            ShadowMasks
    """
    def __init__(self, img_root, pipeline = None) -> None:
        super().__init__(os.path.join(img_root, "ShadowImages"), os.path.join(img_root, "ShadowMasks"), pipeline)

