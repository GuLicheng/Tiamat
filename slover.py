"""
    This is a model framework, you just need provide your
    model and optimizer for your Net
"""
import cv2 as cv
import torch
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
# from configuration import cfg
from configurations import config
from loader.data_loader import TEST_LOADER, TRAIN_LOADER
# from model.test_model import MODEL
from model.model import MODEL

from model.model import Net

MODEL = Net(3)

class Slover:
    """
        ...
    """ 
    def __init__(self, model):
        self.model = model
        self.train_loader = TRAIN_LOADER
        self.test_loader = TEST_LOADER
        self.optimizer = torch.optim.SGD(self.model.parameters(), config.learning_rate)
        self.device = config.device

        # log 
        self.losses = []

    def train(self):
        self.model = self.model.to(self.device)
        optimizer = self.optimizer
        criteon = F.binary_cross_entropy_with_logits
        for epoch in range(200):
            for batch, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criteon(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"OK for {batch} batches")
            print(f"ok for {epoch} epochs and loss is {loss.item()}")
            self.losses.append(loss.item())
            if epoch % 10 == 0:
                self.save_model(f"./result/params{epoch // 10}.pkl", )
                # torch.save(self.model.state_dict(), f"params{epoch // 100}.pkl")
        # model_object.load_state_dict(torch.load('params.pkl'))

    def report_loss(self):
        for idx, loss in enumerate(self.losses):
            print(f"the {idx} loss is {loss}")

    def validate(self):
        pass

    def test(self):
        self.model.eval()
        # pres = 

    def show(self):
        self.model.eval()
        cnt = 3
        for x, y in self.test_loader:
            y = y[0].squeeze_(0).data.numpy()
            x = self.model(x)
            x = x[0].squeeze_(0).data.numpy()
            # ls = [x, y]
            img = np.hstack([x, y])
            # print(y.size())
            # print(y.size())
            cv.imshow("image", img)
            cv.imwrite(f"./image/{cnt}result.png", img * 255)
            # cv.imwrite(f"./image/{cnt}_gt.png", y * 255)

            # cv.waitKey(0)
            if cnt == 0:
                break
            else:
                cnt -= 1            


            
    def save_model(self, file_name: str) -> None:
        print("Saving model paras: " + file_name)
        torch.save(self.model.state_dict(), file_name)


import torchvision.transforms as tt
from PIL import Image
from component.transform import transforms_for_image
if __name__ == '__main__':
    # slover = Slover(MODEL)
    # slover.train()
    net = Net(3)
    net.load_state_dict(torch.load('./result/params19.pkl'))
    net.eval()
    s = Slover(net)
    s.show()

