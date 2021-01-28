import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms

os.path.join("..")

import loss.IoU_Loss

"""
上上级引用
import sys
sys.path.append("...")
"""
DEVICE = torch.device("cuda")


class Net(nn.Module):
    """
        This is a usr-define Net model template, you need provide your model and data_loader here
    """

    def __init__(self, model, data_loader, in_channels=3, out_channels=1, device_type=DEVICE, num_workers=4):
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
        self.model = model

    def forward(self, _x):
        return self.model(_x)

    def train(self):
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        criteon = F.binary_cross_entropy_with_logits
        for epoch in range(500):
            for batch, (x, y) in enumerate(self.data_loader):
                x, y = x.to(self.device_type), y.to(self.device_type)
                logits = self.model(x)
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
