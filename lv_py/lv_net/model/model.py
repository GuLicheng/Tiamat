import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.someblocks.resnet_block import ResNetBlock
from someblocks.resnet_block import ResNetBlock

"""
ForeBackGenerator:
  * The kernel module of ICNet.
    Generate foreground/background co-salient features by using SISMs.
"""
class ForeBackGenerator(nn.Module):
    def __init__(self, H, W):
        super(ForeBackGenerator, self).__init__()
        channel = H * W
        # self.conv = nn.Sequential(nn.Conv2d(channel, 128, 1), Res(128))
        self.conv = ResNetBlock(channel, 128)

    def forward(self, feats: torch.Tensor, SISMs: torch.Tensor):
        N, C, H, W = feats.size()
        HW = H * W
        
        # Resize SISMs to the same size as the input feats.
        SISMs = F.interpolate(SISMs, size=(H, W), mode="bilinear", align_corners=True)
        # NFs: L2-normalized features.
        NFs = F.normalize(feats, dim=1)  # shape=[N, C, H, W]

        def CFM(SIVs: torch.Tensor, NFs: torch.Tensor):
            # Compute correlation maps [Figure 4] between SIVs and pixel-wise feature vectors in NFs by inner product.
            # We implement this process by ``F.conv2d()'', which takes SIVs as 1*1 kernels to convolve NFs.
            correlation_maps = F.conv2d(NFs, weight=SIVs)  # shape=[N, N, H, W]
            
            # Vectorize and normalize correlation maps.
            correlation_maps = F.normalize(correlation_maps.reshape(N, N, HW), dim=2)  # shape=[N, N, HW]
            
            # Compute the weight vectors [Equation 2].
            correlation_matrix = torch.matmul(correlation_maps, correlation_maps.permute(0, 2, 1))  # shape=[N, N, N]
            weight_vectors = correlation_matrix.sum(dim=2).softmax(dim=1)  # shape=[N, N]

            # Fuse correlation maps with the weight vectors to build co-salient attention (CSA) maps.
            CSA_maps = torch.sum(correlation_maps * weight_vectors.view(N, N, 1), dim=1)  # shape=[N, HW]
            
            # Max-min normalize CSA maps.
            min_value = torch.min(CSA_maps, dim=1, keepdim=True)[0]
            max_value = torch.max(CSA_maps, dim=1, keepdim=True)[0]
            CSA_maps = (CSA_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[N, HW]
            CSA_maps = CSA_maps.view(N, 1, H, W)  # shape=[N, 1, H, W]
            return CSA_maps

        def get_SCFs(NFs: torch.Tensor):
            NFs = NFs.view(N, C, HW)  # shape=[N, C, HW]
            SCFs = torch.matmul(NFs.permute(0, 2, 1), NFs).view(N, -1, H, W)  # shape=[N, HW, H, W]
            return SCFs

        # Compute SIVs [Section 3.2, Equation 1]. SISMs will auto expand
        SIVs = F.normalize((NFs * SISMs).mean(dim=(-1, -2)), dim=1).view(N, C, 1, 1)  # shape=[N, C, 1, 1]

        # Compute co-salient attention (CSA) maps [Section 3.3].
        CSA_maps = CFM(SIVs, NFs)  # shape=[N, 1, H, W]

        # Compute self-correlation features (SCFs) [Section 3.4].
        SCFs = get_SCFs(NFs)  # shape=[N, HW, H, W]

        # Rearrange the channel order of SCFs to obtain RSCFs [Section 3.4].
        evidence = CSA_maps.view(N, HW)  # shape=[N, HW]
        indices = torch.argsort(evidence, dim=1, descending=True).view(N, HW, 1, 1).repeat(1, 1, H, W)  # shape=[N, HW, H, W]
        RSCFs = torch.gather(SCFs, dim=1, index=indices)  # shape=[N, HW, H, W]
        cosal_feat = self.conv(RSCFs * CSA_maps)  # shape=[N, 128, H, W]
        return cosal_feat

    @staticmethod
    def test():
        model = ForeBackGenerator(56,56)
        t1 = torch.arange(56*56*3).float().view((1, 3, 56, 56))
        t2 = torch.arange(1*56*56).float().view((1, 1, 56, 56))
        res = model(t1, t2)
        print(res.size())

class ForeBackFusionModule(nn.Module):
    def __init__(self, H, W):
        super(ForeBackFusionModule, self).__init__()
        self.cosal = ForeBackGenerator(H, W)
        self.conv = ResNetBlock(256, 128)

    def forward(self, features: torch.Tensor, SISMs: torch.Tensor) -> torch.Tensor:
        # Get foreground co-saliency features
        fore_cosal_feature = self.cosal(features, SISMs)

        # Get background co-saliency features
        back_cosal_feature = self.cosal(features, 1.0 - SISMs)

        # Fuse foreground and background co-saliency features
        # to generate co-saliency enhance features
        cosal_enhance_features = self.conv(torch.cat([fore_cosal_feature, back_cosal_feature], dim=1))
        return cosal_enhance_features

    @staticmethod
    def test():
        model = ForeBackFusionModule(56, 56)
        feature = torch.arange(56*56*3).float().view((1, 3, 56, 56))
        ssim = torch.arange(56*56*1).float().view((1, 1, 56, 56))
        fuse_feature = model(feature, ssim)
        print(fuse_feature.size())


if __name__ == "__main__":
    ForeBackGenerator.test()
    ForeBackFusionModule.test()

