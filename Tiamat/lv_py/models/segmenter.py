import torch.nn as nn
import torch.nn.functional as F
import torch

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params


    def forward(self, x):

        H, W = x.shape[-2:]

        x = self.encoder(x)

        # remove CLS tokens for decoding
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H // self.encoder.patch_size, W // self.encoder.patch_size))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)

        return masks

    def forward_test(self, im, *args):
        return self.forward_train(im)

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


