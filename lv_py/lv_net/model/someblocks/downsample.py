import torch.nn as nn
import torch


class DownSample(nn.Module):
    """
        For input = (b, c, w, h), the output will be reshaped as (b, c, w//2, h//2)
    """
    def __init__(self, in_channels, batch_normol=True):
        super(DownSample, self).__init__()

        self.bias = not batch_normol

        self.channels = in_channels


        self.layer1 = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            *self._make_layer(1, 1),
            *self._make_layer(3, 1),
            *self._make_layer(3, 2),
        )

        self.layer2 = nn.Sequential(
            self._make_layer(1, 1),
            self._make_layer(3, 2)
        )

        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # change channels
        self.conv = nn.Conv2d(in_channels=self.channels * 3, out_channels=self.channels, kernel_size=1, stride=1, bias=self.bias)


    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)

        x_cat = torch.cat([x1, x2, x3], dim=1)

        return self.conv(x_cat)

    def _make_layer(self, kernel_size, stride):
        if not self.bias:
            BN = nn.BatchNorm2d
        else:
            BN = nn.Identity

        return nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=kernel_size, stride=stride,padding=kernel_size // 2, bias=self.bias),
            BN(self.channels),
            nn.ReLU()
        )




if __name__ == "__main__":
    t = torch.randn(8, 128, 224, 224)
    model = DownSample(128, False)
    print(t.size())
    t = model(t)
    print(t.size())
    # getLayers(model)