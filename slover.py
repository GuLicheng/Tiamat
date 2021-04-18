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
# from loader.secondary_structure_loader import TEST_LOADER, TRAIN_LOADER
from loader.three_level_structure_loader_t import TRAINING_LOADER
from model.transformer.config import Config
config = Config()
# from model.test_model import MODEL

class Slover:
    """
        ...
    """ 
    def __init__(self, model):
        self.model = model
        self.train_loader = TRAINING_LOADER
        self.test_loader = TRAINING_LOADER
        self.optimizer = torch.optim.SGD(self.model.parameters(), config.learning_rate, weight_decay=config.learning_rate / 10)
        self.device = config.device

        # log 
        self.losses = []

    def train(self):
        self.model = self.model.to(self.device)
        optimizer = self.optimizer
        criteon = F.binary_cross_entropy_with_logits
        for epoch in range(5000):
            for batch, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criteon(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"OK for {batch} batches and loss is {loss.item()}")
            print(f"ok for {epoch} epochs and loss is {loss.item()}")
            self.losses.append(loss.item())
            if epoch % 100 == 0:
                self.save_model(f"./result/params{epoch // 10}.pkl")
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
        # with torch.no_grad()
        cnt = 10
        for x, y in self.test_loader:
            x = self.model(x)
            # ls = [x, y]
            print(y.size())
            print(x.size())

            Xs, Ys = torch.split(x, 1, dim=1), torch.split(y, 1, dim=1)
            Xs, Ys = list(Xs), list(Ys)
            for i in range(len(Xs)):
                Xs[i] = Xs[i].view(256, 256).detach().numpy()
                Ys[i] = Ys[i].view(256, 256).detach().numpy()
            pred = np.hstack(Xs)
            gt = np.hstack(Ys)
            res = np.vstack([pred, gt])
            cv.imshow("image", res)
            cv.waitKey(0)
            if cnt == 0:
                break
            else:
                cnt -= 1            


            
    def save_model(self, file_name: str) -> None:
        print("Saving model paras: " + file_name)
        torch.save(self.model.state_dict(), file_name)


# from model.transformer.test_model import Net

from model.transformer.pvt import Net

def debug():
    net = Net()
    net.load_state_dict(torch.load("./result/params400.pkl"))
    net.eval()
    slover = Slover(net)
    slover.show()

def train():
    model = Net()
    slover = Slover(model)
    slover.train()

def run(mode):
    if mode == 1:
        train()
    else:
        debug()

if __name__ == '__main__':
    run(1)



