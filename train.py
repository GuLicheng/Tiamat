from torch._C import device
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim

from model.transformer.test_model import Net
from loader.three_level_structure_loader_t import TRAINING_LOADER

def train():
    model = Net()
    model.train()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criteon = F.binary_cross_entropy_with_logits
    losses = []
    for epoch in range(500):
        for batch, (x, y) in enumerate(TRAINING_LOADER):
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

train()