import torch
import torch.nn as nn
from typing import Optional

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1)

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size = 1)
        else:
            self.shortcut = nn.Identity()
        
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(self.time_act(t))[:, :, None, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None):
        super().__init__()

        if d_k is None:
            d_k = n_channels
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        
        batch_size, n_channels, sagittal, coronal, axial = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)

        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim = -1)
        attn = torch.einsum('b i h d, b j h d -> b i j h', q, k) * self.scale
        attn = attn.softmax(dim = 2)

        res = torch.einsum('b i j h, b j h d -> b i h d', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, sagittal, coronal, axial)
        return res

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int, dropout: float, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, n_groups, dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int, dropout: float, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, n_groups, dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, 4, 2, 1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        return self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self, in_channels: int, time_channels: int,
                 n_groups: int, dropout: float, has_attn: bool = False):
        super().__init__()
        self.down = Downsample(in_channels)
        self.res1 = ResidualBlock(in_channels, in_channels * 2, time_channels, n_groups, dropout)
        self.se = SEBlock(in_channels * 2, in_channels * 2 // n_groups)
        if has_attn:
            self.attn = AttentionBlock(in_channels * 2)
        else:
            self.attn = nn.Identity()
        self.res2 = ResidualBlock(in_channels * 2, in_channels, time_channels, n_groups, dropout)
        self.up = Upsample(in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.down(x, t)
        x = self.res1(x, t)
        x = self.se(x)
        x = self.attn(x)
        x = self.res2(x, t)
        x = self.up(x, t)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels: int, r: int):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.SiLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        assert x.ndim == 5
        # z = self.squeeze(x).squeeze(-3, -2, -1)
        z = self.squeeze(x).squeeze(-1).squeeze(-1).squeeze(-1)
        z = self.excitation(z)
        x = x * z[:, :, None, None, None]
        return x