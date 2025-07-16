import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Tuple, Union, List

from models.UNet3D.utils import *
from models.UNet3D.modules import *

class UNet(nn.Module):
    def __init__(self,
                 age_embed: bool,
                 age_embed_dim: int,
                 n_channels: int,
                 n_groups: int,
                 n_blocks: int,
                 ch_mults: Union[Tuple[int, ...], List[int]],
                 is_attn: Union[Tuple[bool, ...], List[int]],
                 dropout: float,
                 use_checkpoint: bool,
                 **kwargs):
        super().__init__()

        assert np.mod(n_channels, n_groups) == 0
        assert len(ch_mults) == len(is_attn)

        self.age_embed = age_embed
        self.age_embed_dim = age_embed_dim
        self.n_channels = n_channels
        self.n_groups = n_groups
        self.n_blocks = n_blocks
        self.use_checkpoint = use_checkpoint

        n_resolutions = len(ch_mults)
        
        self.time_emb = TimeEmbedding(n_channels * 4)
        
        if age_embed:
            assert age_embed_dim is not None
            self.age_diff_emb = nn.Sequential(
                AgeEmbedding(age_embed_dim),
                nn.Linear(age_embed_dim, n_channels * 4),
                nn.SiLU(),
                nn.Linear(n_channels * 4, n_channels * 4)
            )
            self.age_cond_emb = nn.Sequential(
                AgeEmbedding(age_embed_dim),
                nn.Linear(age_embed_dim, n_channels * 4),
                nn.SiLU(),
                nn.Linear(n_channels * 4, n_channels * 4)
            )

        self.in_conv = nn.Conv3d(1, n_channels, kernel_size = 1)

        self.in_concat_conv = nn.Conv3d(n_channels + 1, n_channels, kernel_size = 3, padding = 1)
        # self.in_res_conv = ResidualBlock(n_channels, n_channels, n_channels * 4, n_groups, dropout, False)
        
        in_channels = n_channels
        down = []
        
        for i in range(n_resolutions):
            
            out_channels = in_channels * ch_mults[i]
            
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, n_groups, dropout, is_attn[i]))
                in_channels = out_channels
            
            if i < n_resolutions - 1:
                down.append(Downsample(out_channels))
        
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_channels * 4, n_groups, dropout, False)

        up = []

        for i in reversed(range(n_resolutions)):
            
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels + out_channels, out_channels, n_channels * 4, n_groups, dropout, is_attn[i]))
            
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels + out_channels, out_channels, n_channels * 4, n_groups, dropout, is_attn[i]))
            in_channels = out_channels

            if i > 0:
                up.append(Upsample(out_channels))

        self.up = nn.ModuleList(up)

        self.out_concat_conv = nn.Conv3d(n_channels + 1, n_channels, kernel_size = 3, padding = 1)
        # self.out_res_conv = ResidualBlock(n_channels, n_channels, n_channels * 4, n_groups, dropout, False)

        self.out_conv = nn.Conv3d(n_channels, 1, kernel_size = 1)

    def forward(self, x, age_diff, cond, cond_mask, t):
        b, _, d_, h_, w_ = x.shape
        assert (b,) == age_diff.shape == cond_mask.shape

        img_cond = cond['img_cond']
        img_cond *= cond_mask[:, None, None, None, None]
        
        t = self._maybe_checkpoint(self.time_emb, t)
        
        if self.age_embed:
            age_cond = cond['age_cond']
            age_diff_embedding = self._maybe_checkpoint(self.age_diff_emb, age_diff)
            age_cond_embedding = self._maybe_checkpoint(self.age_cond_emb, age_cond)
            age_embedding = age_diff_embedding + age_cond_embedding
            t = t + age_embedding * cond_mask[:, None]

        x = self._maybe_checkpoint(self.in_conv, x)
        x = torch.cat((x, img_cond), 1)
        x = self._maybe_checkpoint(self.in_concat_conv, x)
        # x = self.in_res_conv(x, t)

        h = [x]
        
        for m in self.down:
            x = self._maybe_checkpoint(m, x, t)
            h.append(x)
        
        x = self._maybe_checkpoint(self.middle, x, t)
        
        for m in self.up:
            if isinstance(m, Upsample):
                x = self._maybe_checkpoint(m, x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), 1)
                x = self._maybe_checkpoint(m, x, t)
        
        x = torch.cat((x, img_cond), 1)
        x = self._maybe_checkpoint(self.out_concat_conv, x)
        # x = self.out_res_conv(x, t)
        out = self._maybe_checkpoint(self.out_conv, x)
        return out
    
    def _maybe_checkpoint(self, module, *inputs):
        # checkpointing except for modules related to timestep- and age-embedding
        if self.use_checkpoint:
            return checkpoint(module, *inputs, use_reentrant = False)
        else:
            return module(*inputs)