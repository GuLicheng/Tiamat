import torch.nn as nn

configs = {
    # 13 convolutions and 5 maxpools
    "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def vgg16(config=configs["vgg16"], in_channels=3, batch_norm=True):
    """
        Create a vgg16 backbone, the finally out_channels will be the last number in configurations
        Parameters:
            config: see in configs(default for vgg16)
            in_channels: the in_channels of your sample(default for 3-RGB)
            batch_norm: whether batch normalization or not
        Return:
            a list of layer consist of layer in vgg16(with 13 convolution layers and 5 maxpool layers)
    """
    layers = []
    _in_channels = in_channels
    assert config.__len__() == 18
    for out in config:
        if out == 'M':
            # add maxpool
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers.extend([nn.MaxPool2d(kernel_size=2, stride=2)])
        else:
            # out is out_channels
            conv2d = nn.Conv2d(_in_channels, out, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(out), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            _in_channels = out
    return layers


VGG16 = vgg16()




def test_for_vgg16() -> None:
    net = vgg16()
    for layer in net:
        print(layer)

if __name__ == '__main__':
    test_for_vgg16()
