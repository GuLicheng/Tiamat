"""
    This is a model framework, you just need provide your
"""
import torch

from configuration import cfg
import torch.nn.functional as F


class Slover:
    """
    """

    def __init__(self,
                 model=cfg["model"],
                 train_loader=cfg["train_loader"],
                 test_loader=cfg["test_loader"],
                 optimizer=cfg["optimizer"],
                 device=cfg["device"]
                 ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device

    def train(self):
        self.model = self.model.to(self.device)
        optimizer = self.optimizer
        criteon = F.binary_cross_entropy_with_logits
        for epoch in range(10):
            for batch, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criteon(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"OK for {batch} batches")
            print(f"ok for {epoch} epochs and loss is {loss.item()}")
            if epoch % 100 == 0:
                self.save_model(f"./result/params{epoch // 100}.pkl", )
                # torch.save(self.model.state_dict(), f"params{epoch // 100}.pkl")
        # model_object.load_state_dict(torch.load('params.pkl'))

    def validate(self):
        pass

    def test(self):
        pass

    def save_model(self, file_name: str) -> None:
        print("Saving model paras: " + file_name)
        torch.save(self.model.state_dict(), file_name)


if __name__ == '__main__':
    slover = Slover()
    slover.train()
