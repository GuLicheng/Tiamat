import torch.nn as nn
import torch
import math
import torch.optim

def save_model(dest: str, **kwargs):
    
    def get_state_dict(x):
        if hasattr(x, "state_dict"):
            return x.state_dict()
        return x

    checkpoint = { }
    for k, v in kwargs.items():
        checkpoint[k] = get_state_dict(v)
        print(f"Successfully get {k}")
    torch.save(checkpoint, dest)
    print(f"Successfully save model at {dest}")

class WeightsInitializer:

    @staticmethod
    def xavier(m: nn.Module):
        assert isinstance(m, nn.Module)
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, val=0.)


if __name__ == "__main__":
    conv = nn.Conv2d(3, 3, 1)
    WeightsInitializer.xavier(conv)
    SGD = torch.optim.Adam(conv.parameters(), lr=0.1)
    save_model("./a.pth", model=conv, optimizer=SGD, epoch=1)