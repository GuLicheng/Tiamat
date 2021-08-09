import torch
import torch.nn as nn
import numpy as np
import timm

class PatchEmbeedding(nn.Module):
    """
        Split image into patches and embed them
    """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2 # img_size ** 2 // patch_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        return x

class Attention(nn.Module):
    
    def __init__(self, dim, n_heads = 12, qkv_bias = True, attn_p = 0., proj_p = 0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** (-0.5)  # softmax / (sqrt(...))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape # token -> img spatial pixel
        if dim != self.dim:
            raise ValueError()
        qkv = self.qkv(x)

        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, n_samples, n_heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (n_samples, n_heads, n_patches, head_dim)
        qk = q @ k.transpose(-1, -2) * self.scale
        attn = qk.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weight_avg = attn @ v
        weight_avg = weight_avg.transpose(1, 2)
        weight_avg = weight_avg.flatten(2) # (n_samples, n_patches, dim)
        x = self.proj(weight_avg)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, p = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_states = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_states, out_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) 
        # x -> ref -> ptr -> tensor1, self.attn -> tensor2  
        # x = x + ... -> *x = tensor1 + tensor2  | memory -> (1, 2, *x)
        # x += -> overload += operator  | memory (tensor2, *x) -> backward
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):

    def __init__(self, img_size=384, patch_size=16, in_chans=3, n_classes=1000, 
                embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, 
                p=0, attn_p=0.):
        super().__init__()
        self.patch_embed = PatchEmbeedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))

        self.pos_drop = nn.Dropout(p)
        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio, qkv_bias, p, attn_p)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes) # self.classifier

    def forward(self, x):
        n_samples = x.size(0)
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        ) 
        # (n_samples, 1, embed_dim) (49 = 7 x 7)
        # (n_samples, 1 + 49 = 50, embed_dim)

        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0] # just the cls token
        x = self.head(cls_token_final)
        return x


# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)

model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
print(type(model_official))

custom_config = {
        "img_size": 384,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
}

def main():
    model_custom = VisionTransformer(**custom_config)
    model_custom.eval()


    for (n_o, p_o), (n_c, p_c) in zip(
            model_official.named_parameters(), model_custom.named_parameters()
    ):
        assert p_o.numel() == p_c.numel(), f"Expected {p_o.numel()}, but got {p_c.numel()}"
        print(f"{n_o} | {n_c}")

        p_c.data[:] = p_o.data

        assert_tensors_equal(p_c.data, p_o.data)

    inp = torch.rand(1, 3, 384, 384)
    res_c = model_custom(inp)
    res_o = model_official(inp)

    # Asserts
    assert get_n_params(model_custom) == get_n_params(model_official)
    assert_tensors_equal(res_c, res_o)

    # Save custom model
    torch.save(model_custom, "model.pth")

if __name__ == "__main__":
    # t = torch.randn(1, 3, 384, 384)
    # model = VisionTransformer()
    # res = model(t)
    # print(res.size())
    main()