import os

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.autograd import Variable
os.path.join("..")

import loss.IoU_Loss

from uitility.utils import add_printer_into_sequence
from uitility.utils import Printer1


"""
上上级引用
import sys
sys.path.append("...")
"""
device = torch.device("cuda")


class Block(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Block, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.model(x)


class Net(nn.Module):
    """
        This is a usr-define Net model based on vgg16
    """

    def __init__(self, data_loader, in_channels, out_channels, device_type=device, num_workers=4):
        """
            Parameters:
                in_channels:
                out_channels:
                device_type: cuda or cpu or other modes
                num_workers: the number of thread
        """
        super(Net, self).__init__()
        self.data_loader = data_loader
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device_type = device_type
        self.num_workers = num_workers
        self.IoU_Loss = loss.IoU_Loss.LossFunctionFactory.IoU_Loss_for_image()

        """
            Define the structure of your model here
        """
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1)
        self.block1 = Block(in_channels=128, out_channels=256)
        self.block2 = Block(in_channels=256, out_channels=512)
        self.block3 = Block(in_channels=512, out_channels=512)
        self.block4 = Block(in_channels=512, out_channels=512)

        self.conv1_for_out1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)
        self.conv2_for_out2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv3_for_out3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.tt = nn.Conv2d(512, 512, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        down_out1 = self.block1(x)  # 224
        down_out2 = self.block2(down_out1)  # 112
        down_out3 = self.block3(down_out2)  # 56
        down_out4 = self.block4(down_out3)  # 28

        t1 = self.tt(down_out4)
        t1 = self.upsample(t1)
        t1 += down_out3
        t1 = self.conv3_for_out3(t1)
        t1 = self.upsample(t1)
        t1 += down_out2
        t1 = self.conv2_for_out2(t1)
        t1 = self.upsample(t1)
        t1 += down_out1
        t1 = self.conv1_for_out1(t1)
        print(t1.shape)
        return t1


    def train(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        criteon = F.binary_cross_entropy_with_logits
        for epoch in range(500):
            for batch, (x, y) in enumerate(self.data_loader):
                x, y = x.to(device), y.to(device)
                logits = self.forward(x)
                loss = 0.7 * criteon(logits, y) + 0.3 * self.IoU_Loss(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"OK for {batch} batches")
            print(f"ok for {epoch} epochs and loss is {loss.item()}")
        if epoch % 100 == 0:
            torch.save(self.state_dict(), f"params{epoch // 100}.pkl")
        # model_object.load_state_dict(torch.load('params.pkl'))


    def test_a_sample(self):
        trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img = Image.open(r"D:\MY\DataSet\DUTS\DUTS-TR\DUTS-TR-Image\ILSVRC2012_test_00000004.jpg")
        shape = img.size
        img.show()
        img = trans(img) - torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255
        print("after transform...")
        with torch.no_grad():
            img = Variable(img.unsqueeze(0))
        prob = self.model(img)
        prob = (prob.cpu().data[0][0].numpy() * 225).astype(np.uint8)
        print("after reshaping")
        p_img = Image.fromarray(prob, mode='L').resize(shape)
        p_img.show()




