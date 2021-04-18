import torch
import numpy as np
import cv2 as cv
import torchvision.transforms.functional

class Flip:
    def __init__(self) -> None:
        pass

    def __call__(self, tensor: torch.Tensor):
        assert tensor.size().__len__() == 4
        flip1 = torch.flip(tensor, dims=(2,3))
        flip2 = torch.flip(tensor, dims=(3,))
        flip3 = torch.flip(tensor, dims=(2,))
        return flip1, flip2, flip3

    @staticmethod
    def test():
        tensor = torch.arange(24).view((1, 2, 3, 4))
        tensors = Flip()(tensor)
        for t in tensors:
            print(t.size(), '\n', t)
    
    @staticmethod
    def test2():
        tensors = np.arange(0, 24).reshape((1, 2, 3, 4))
        t1 = np.flip(tensors, 1)  # 水平
        t2 = np.flip(tensors, 0)  # 竖直
        t3 = np.flip(tensors, -1) # 水平 + 竖直
        print(t1)
        print(t2)
        print(t3)


if __name__ == "__main__":
    Flip.test2()
    Flip.test()

"""
[[[[12 13 14 15]   
   [16 17 18 19]   
   [20 21 22 23]]  

  [[ 0  1  2  3]   
   [ 4  5  6  7]   
   [ 8  9 10 11]]]]

[[[[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]]

  [[12 13 14 15]
   [16 17 18 19]
   [20 21 22 23]]]]

[[[[ 3  2  1  0]
   [ 7  6  5  4]
   [11 10  9  8]]

  [[15 14 13 12]
   [19 18 17 16]
   [23 22 21 20]]]]
"""