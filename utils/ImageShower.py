import cv2 as cv
import numpy as np
from typing import *
 

class ImageShower:

    def __init__(self, windowname: str, paths: List[str], img_size: Tuple[int, int] = (900, 700)) -> None:
        self.windowname = windowname
        self.paths = paths
        self.size = img_size

    def show(self, size: Tuple[int, int], gap_size: int = 3) -> None:
        """(width, height)"""
        assert size[0] * size[1] == len(self.paths)
        row, col = size
        images = [self.__read_image(self.paths[i]) for i in range(len(self.paths))]
        height = row * gap_size + row * self.size[1]
        width = col * gap_size + col * self.size[0]
        screen = np.full((height, width, 3), 255, np.uint8)
        cv.namedWindow(self.windowname, flags=0)
        for idx, image in enumerate(images):
            x, y = self.__get_position(size, idx, gap_size)
            print(f"(x, y) = ({x}, {y})")
            for i in range(self.size[1]):
                for j in range(self.size[0]):
                    screen[i + x, j + y] = image[i, j]

        cv.imshow(self.windowname, screen)
        cv.waitKey(0)
        
    
    def __read_image(self, img_name: str) -> np.ndarray :
        flag = cv.IMREAD_COLOR if img_name.endswith(".jpg") else cv.IMREAD_GRAYSCALE
        img = cv.imread(img_name, flag)
        if img.shape[2] == 1:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        img = cv.resize(img, self.size)
        return img

    def __get_position(self, size: Tuple[int, int], idx: int, gap_size: int) -> Tuple[int, int]:
        x, y = idx // size[1], idx % size[1]
        offset_x, offset_y = x * self.size[1] + x * gap_size, y * self.size[0] + y * gap_size
        return (x + offset_x, y + offset_y)

if __name__ == "__main__":
    root = [r"C:\Users\Administrator\Pictures\Saved Pictures\01.jpg",
            r"C:\Users\Administrator\Pictures\Saved Pictures\wallhaven-x86p8l.jpg"]
    handle = ImageShower("", root)
    handle.show((1, 2))