import torch
import torch.nn as nn
from einops import rearrange, einsum
import math

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.feedforward(x)

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout):
        super().__init__()
        hidden_dim = hidden_dim or dim
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()

        dim_head = dim_head or dim

        inner_dim = dim_head * heads
        # project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) # if project_out else nn.Identity()

    def forward(self, x, pad_mask):
        assert x.ndim == 3
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        dots = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        if pad_mask is not None:
            dots = dots.masked_fill(pad_mask == 0, -1e9)

        attn = dots.softmax(dim = -1)
        
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()

        dim_head = dim_head or dim

        inner_dim = dim_head * heads
        # project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) # if project_out else nn.Identity()

    def forward(self, x_query, y_key, y_value, pad_mask):
        assert x_query.ndim == y_key.ndim == 3
        assert y_key.shape == y_value.shape
        h = self.heads

        x_query = self.to_q(x_query)
        y_key = self.to_k(y_key)
        y_value = self.to_v(y_value)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (x_query, y_key, y_value))
        
        dots = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        if pad_mask is not None:
            dots = dots.masked_fill(pad_mask == 0, -1e9)

        attn = dots.softmax(dim = -1)
        
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, n_channels, r):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(n_channels, n_channels // r),
            nn.GELU(),
            nn.Linear(n_channels // r, n_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, d, h, w = x.shape
        z = self.squeeze(x).squeeze(-3, -2, -1) # (b, d)
        z = self.excitation(z)
        x = x * z[:, :, None, None, None]
        return x

class DecodeBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, n_groups, expand_r, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.expand_r = expand_r

        self.expand_ksp = {1 : (3, 1, 1), 2 : (4, 2, 1), 3 : (5, 3, 1)}.get(expand_r, (3, 1, 1))

        self.expand = nn.Sequential(
            nn.GroupNorm(1, in_channels),  # LayerNorm; instead of nn.GroupNorm(n_groups, in_channels),
            nn.GELU(),
            # nn.Dropout(dropout),  # deprecated; dropout seriously hinders decoding
            # SEBlock(in_channels, in_channels // n_groups),  # deprecated; works but found unnecessary
            nn.ConvTranspose3d(in_channels, mid_channels, *self.expand_ksp),
        )
        
        self.retain = nn.Sequential(
            nn.GroupNorm(1, mid_channels),  # LayerNorm; instead of nn.GroupNorm(n_groups, mid_channels),
            nn.GELU(),
            # nn.Dropout(dropout),  # deprecated; dropout seriously hinders decoding
            # SEBlock(mid_channels, mid_channels // n_groups),  # deprecated; works but found unnecessary
            nn.Conv3d(mid_channels, out_channels, *(3, 1, 1)),
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.retain(x)
        return x

class AgeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, t):
        half_dim = self.n_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim = 1)
        return emb