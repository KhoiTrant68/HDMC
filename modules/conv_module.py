import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

class LayerNorm2d(nn.Module):
    """
    LayerNorm that supports inputs of shape (N, C, H, W).
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class ConvBottleneckBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 3, drop_path=0.0):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)
        )
        layers = []
        for _ in range(num_layers):
            layers.append(ConvNeXtBlock(out_ch, drop_path=drop_path))
        self.res_blocks = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.downsample(x)
        out = self.res_blocks(out)
        return out

class ConvBottleneckBlockWithUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 3, drop_path=0.0):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ConvNeXtBlock(in_ch, drop_path=drop_path))
        self.res_blocks = nn.Sequential(*layers)
        
        self.upsample = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), # bias is False here
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.res_blocks(x)
        out = self.upsample(out)
        return out