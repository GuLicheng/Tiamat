import torch

from models.segmenter import Segmenter
from backbones.vit import vit_base, vit_small, vit_tiny
from decode_head.segmenter_head import MaskTransformer, DecoderLinear

def segmenter():
    backbone = vit_small()
    decode_head = MaskTransformer()
    model = Segmenter(backbone, decode_head)
    return model


if __name__ == "__main__":


    input_tensor = torch.randn(2, 3, 500, 375)

    model = segmenter()

    result = model(input_tensor)

    print(result.shape)