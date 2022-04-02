import cv2 as cv
import numpy as np
from linq import make_sequence

from functools import singledispatchmethod, singledispatch

from PIL import Image
import torch

def resize(img: np.ndarray, image_size: int):
    mode = cv.INTER_NEAREST if len(img.shape) == 2 else cv.INTER_LINEAR  
    return cv.resize(img, dsize=(image_size, image_size), interpolation=mode)

def normalize(img: np.ndarray, eps: float = 1e-5):
    min_val, max_val = img.min(), img.max()
    img = (img - min_val) / (max_val - min_val + eps)
    return img

def to3channel(img: np.ndarray):
    if len(img.shape) == 3:
        assert img.shape[2] in [1, 3]
        return img
    img = (normalize(img) * 255).astype(np.uint8)
    img = cv.applyColorMap(img, cv.COLORMAP_JET)
    return img

def convert_rgb2bgr(img: np.ndarray):
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)

@singledispatch
def read_image_from(*args): ...

@read_image_from.register(str)
def read_image_from_file(filename: str, mode: int = cv.IMREAD_ANYCOLOR):
    return cv.imread(filename, mode)

@read_image_from.register(Image.Image)
def read_image_from_PIL(PIL_image: Image.Image):
    return convert_rgb2bgr(np.array(PIL_image))

@read_image_from.register(torch.Tensor)
def read_mask_from_tensor(tensor: torch.Tensor):
    assert len(tensor.shape) == 2
    return tensor.numpy()

class ImageManager:

    def __init__(self, **kwargs) -> None:
        self.images = make_sequence([])
        self.cfg = dict(channel=3, image_size=480)

        self.cfg.update(**kwargs)

    def hstack(self, **kwargs):
        kwargs.update(self.cfg)
        image_size = kwargs["image_size"]
        channel = kwargs["channel"]
        assert isinstance(image_size, int) and channel in [3, 4]

        images = self.images.select(lambda image: resize(image, image_size)).select(to3channel).to_list()
        image = np.hstack(images)
        return image

    def read_images(self, *args):
        self.images.concat([read_image_from(arg) for arg in args])
        return self

    def imshow(self, winname: str = "window", **kwargs):
        
        images = self.hstack(**kwargs)
        cv.imshow(winname, images)
        cv.waitKey(0)
        cv.destroyWindow(winname)
        return self

    def imwrite(self, dest: str, **kwargs):
        kwargs.update(self.cfg)
        images = self.hstack(**kwargs)
        cv.imwrite(dest, images)
        return self


if __name__ == "__main__":
    img1 = r"C:\Users\dell\Pictures\Saved Pictures\t01d70f03f64fdb7d37.jpg"
    img2 = r"C:\Users\dell\Pictures\Saved Pictures\t01a15f1966b85bae1b.jpg"

    # img2 = Image.open(img2).convert("RGB")

    image_manager = ImageManager()
    image_manager.read_images(img1, img2).imwrite("b.png", image_size=480).imwrite("a.png", image_size=240)

