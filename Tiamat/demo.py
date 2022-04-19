from typing import Counter
import cv2 as cv
import numpy as np
from tqdm import tqdm
import cpp


def label_expand(superpixel: np.ndarray, label: np.ndarray, superpixel_num: np.ndarray, ignore_index: int = 255):
    """
        ignore_index: unlabeled pixel
    """
    pred = np.full_like(superpixel, ignore_index)
    for i in range(0, superpixel_num):
        mask = (superpixel == i) & (label != ignore_index) 
        selected = label[mask]
        counter = Counter(selected.flatten())
        pred[superpixel == i] = max(counter, key=lambda x: counter[x]) if counter else ignore_index
    return np.uint8(pred)

def test_superpixel():
    image = cv.imread("images/2007_000032.jpg")
    scribble = cv.imread("images/2007_000032.png", cv.IMREAD_GRAYSCALE)


    slic = cv.ximgproc.createSuperpixelSLIC(image, region_size=10, ruler=20) 
    slic.iterate(50)
    label_slic = slic.getLabels()
    number_slic = slic.getNumberOfSuperpixels()
    superpixel2 = cpp.label_expand(label_slic, scribble, number_slic, 255)
    superpixel1 = label_expand(label_slic, scribble, number_slic, ignore_index=255)

    from lv_py.dataset.pascalvoc import PascalVoc

    img1 = PascalVoc.decode_segmap(superpixel1)
    img2 = PascalVoc.decode_segmap(superpixel2)

    img = np.hstack([image, img1, img2])

    cv.imshow("", img)
    cv.waitKey(0)


if __name__ == "__main__":

    test_superpixel()