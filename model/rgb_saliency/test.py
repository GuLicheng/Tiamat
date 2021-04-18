import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from model.someblocks.resnet_block import ResNetBlock
from model.someblocks.saliencymap_generator import SaliencyMapGenerator
from loader.secondary_structure_loader import SecondDirectoryStructureDataLoader

class TestNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.res1_1 = ResNetBlock(3, 64, True)
        self.res1_2 = ResNetBlock(64, 64, True)
        self.res2_1 = ResNetBlock(64, 128, True)
        self.res2_2 = ResNetBlock(128, 128, True)
        self.res3_1 = ResNetBlock(128, 256, True)        
        self.res3_2 = ResNetBlock(256, 256, True)        
        self.res3_3 = ResNetBlock(256, 256, True)        
        self.res4_1 = ResNetBlock(256, 512, True)        
        self.res4_2 = ResNetBlock(512, 512, True)        
        self.res4_3 = ResNetBlock(512, 512, True)  
        self.res5_1 = ResNetBlock(512, 512, True)
        self.res5_2 = ResNetBlock(512, 512, True)
        self.res5_3 = ResNetBlock(512, 512, True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()
        self.pred = SaliencyMapGenerator(512)  

        self.res6 = ResNetBlock(512, 256)
        self.res7 = ResNetBlock(256, 128)  
        self.res8 = ResNetBlock(128, 64)  

        self.layers1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4), # 112 * 112, 512
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4), #224 * 224
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.maxpool(x)
        y1 = x

        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.maxpool(x)
        y2 = x

        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res3_3(x)
        x = self.maxpool(x)
        y3 = x

        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        x = self.maxpool(x)
        y4 = x

        x = self.res5_1(x)
        x = self.res5_2(x)
        x = self.res5_3(x)
        x = self.maxpool(x)

        x = self.layers1(x)
        x = F.interpolate(x, scale_factor=2)
        return x

transforms_for_image: transforms.Compose = transforms.Compose([
    lambda x: Image.open(x).convert("RGB"),
    # ImageOpenConvertRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# pre-process for label
transforms_for_label: transforms.Compose = transforms.Compose([
    lambda x: Image.open(x).convert("1"),
    # ImageOpenConvert1(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

loader = SecondDirectoryStructureDataLoader(
    root=r"G:/DataSet/SOD/",
    dirs=("image", "gt"),
    tfs=(transforms_for_image, transforms_for_label),
    suffixes=(".jpg", ".png"),
    mode="train"
)

loader = DataLoader(
    loader, batch_size=16, num_workers=0
)

def train():
    model = TestNet()
    # model.load_state_dict(torch.load(r"D:\MY\SOD\Tiamat\result\params4.pkl"))
    model.train()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criteon = F.binary_cross_entropy_with_logits
    losses = []
    for epoch in range(500):
        for batch, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"OK for {batch} batches and loss is {loss.item()}")
        # losses.append(loss.item())
        print(f"ok for {epoch} epochs")
        # self.losses.append(loss.item())
        if epoch % 10 == 0:
            print("Saving model paras: " + f"params{epoch // 100}.pkl")
            torch.save(model.state_dict(), f"./result/params{epoch // 100}.pkl")
            # torch.save(self.model.state_dict(), f"params{epoch // 100}.pkl")
    # model_object.load_state_dict(torch.load('params.pkl'))
    torch.save(model.state_dict(), f"./result/params_finally.pkl")

def test():
    model = TestNet()
    model.load_state_dict(torch.load(r"D:\MY\SOD\Tiamat\result\params4.pkl"))
    model.eval()
    # with torch.no_grad()
    cnt = 10
    with torch.no_grad():
        for x, y in loader:
            x: torch.Tensor = model(x)
            Xs = list(torch.split(x, 1, dim=0))
            Ys = list(torch.split(y, 1, dim=0))
            for i in range(len(Xs)):
                Xs[i] = Xs[i].view(224, 224).detach().numpy()
                Ys[i] = Ys[i].view(224, 224).detach().numpy()
            Xs = np.hstack(Xs)
            Ys = np.hstack(Ys)
            images = np.vstack([Xs, Ys])
            cv.imshow("xxx", images)
            cv.waitKey(0)
            exit()

if __name__ == "__main__":
    # test()
    train()