from component.transform import *
from model.vgg16 import VGG16
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(*VGG16)
        self.reader = transforms_for_image

    def forward(self, path):
        x = self.reader(path)
        x = self.layers(x)
        return x

def main():
    model = Model()
    res = model(r"C:\Users\dell\Pictures\Saved Pictures\t01a15f1966b85bae1b.jpg")
    print(res)

if __name__ == "__main__":
    main()