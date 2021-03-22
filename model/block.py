import torch
import torch.nn as nn
import torch.nn.init


class MirrorPaddingConv2d(nn.Module):
    """ 
        mirror padding used to replace padding
        
        >>> self.conv = nn.Conv2d(...)
        >>> self.conv = MirrorPaddingConv(...)
    """
    def __init__(self, in_channel, out_channel):
        super(MirrorPaddingConv2d).__init__()
        self.padding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        return self.relu(x)


class SelfAttentionBlock(nn.Module):
    """
        A simple self-attention block, which would not change the size of image.
        For input(b, c, w, h), the output is (b, c, w, h)
    """
    def __init__(self, in_channels, out_channels, radio=8):
        assert in_channels >= 64, "The in_channels must greater equal than 64"
        assert in_channels % radio == 0, "The in_channels must be divided by radio"

        super(SelfAttentionBlock, self).__init__()

        # basic parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.middle_channels = in_channels // radio

        # Q, K, V matrix
        self.node_q = self._conv3x3()
        self.node_k = self._conv3x3()
        self.node_v = self._conv3x3()

        self.softmax = nn.Softmax(dim=1) # channel

        # change channels to out_channels
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=self.middle_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.gamma = nn.Parameter(torch.ones(1)) # weight of self-attention

    def forward(self, x):
        """
        paras:
            x: input tensor, with size (b, c, h, w)

        return:
            out: self attention feature maps with size (b, c, h, w)
        """
        # b, c, h, w = x.size()
        node_q = self.node_q(x)
        node_k = self.node_k(x)
        node_v = self.node_v(x)

        # for debugging, disable this when training
        # assert node_q.size() == node_k.size() == node_v.size()

        # our q, k, v will change the input channels
        b, c, h, w = node_k.size()

        # transpose
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)  # b, c, w, h
        node_q = node_q.view(b, c, -1)                   # b, c, h, w
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)  # b, c, w, h

        """
            for n = w * h, c = c
            Y = Softmax(Q * K^T) * V -> O(n * n * c)
            Y = 
        """
        # A = k * q
        # AV = k * q * v
        # AVW = k * (q * v) * w

        AV = torch.bmm(node_q, node_v) # Q * V -> b, c, h, h
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)  # Q * K * V -> b, c, w, h
        AVW = AV.transpose(1, 2).contiguous().view(b, c, h, w) # -> b, c, h, w
        AVW = self.out(AVW)
        out = AVW * self.gamma + x

        return out

    def _conv3x3(self):
        """
            generate a simple 3x3 convolution and init it
        """
        conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.middle_channels, kernel_size=1, stride=1, bias=False)
        nn.init.xavier_normal_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()
        return nn.Sequential(conv, nn.BatchNorm2d(self.middle_channels))

if __name__ == "__main__":
    t = torch.randn(8, 64, 224, 224)
    self_attention = SelfAttentionBlock(64, 64)
    out = self_attention(t)
    print(out)