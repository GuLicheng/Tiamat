import visdom
import torch
import random


def main():
    tensor = torch.Tensor([random.randint(0, 5) for i in range(10)])
    viz = visdom.Visdom()
    viz.line(tensor)

if __name__ == '__main__':
    main()