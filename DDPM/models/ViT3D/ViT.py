import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath

from models.ViT3D.utils import *
from models.ViT3D.modules import *

class SelfAttnTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio, dropout, path_dropout, use_checkpoint):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        mlp_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim, dim, dropout)
        # self.mlp = nn.Sequential(
        #     FeedForward(dim, mlp_dim, dropout),
        #     FeedForward(mlp_dim, dim, dropout),
        # )
        self.drop_path = DropPath(path_dropout) if path_dropout > 0. else nn.Identity()

    def forward_part1(self, x, pad_mask):
        x = self.drop_path(self.attn(self.norm1(x), pad_mask)) + x  # v1 (Pre-Norm)
        # x = self.drop_path(self.norm1(self.attn(x, pad_mask))) + x  # v2 (Post-Norm)
        # x = self.drop_path(self.norm1(self.attn(x, pad_mask) + x))  # v3 (Add-and-Norm)
        return x
    
    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x))) + x  # v1 (Pre-Norm)
        # x = self.drop_path(self.norm2(self.mlp(x))) + x  # v2 (Post-Norm)
        # x = self.drop_path(self.norm2(self.mlp(x) + x))  # v3 (Add-and-Norm)
        return x

    def forward(self, x, pad_mask):
        B_, N, C = x.shape
        x = self._maybe_checkpoint(self.forward_part1, x, pad_mask)
        x = self._maybe_checkpoint(self.forward_part2, x)
        return x
    
    def _maybe_checkpoint(self, module, *inputs):
        if self.use_checkpoint:
            return checkpoint(module, *inputs, use_reentrant = False)
        else:
            return module(*inputs)

class CrossAttnTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio, dropout, path_dropout, use_checkpoint):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        mlp_dim = int(dim * mlp_ratio)

        self.norm1_x = nn.LayerNorm(dim)  # v1 (Pre-Norm)
        self.norm1_y = nn.LayerNorm(dim)  # v1 (Pre-Norm)
        # self.norm1 = nn.LayerNorm(dim)  # v2 (Post-Norm), v3 (Add-and-Norm)
        self.attn = CrossAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim, dim, dropout)
        # self.mlp = nn.Sequential(
        #     FeedForward(dim, mlp_dim, dropout),
        #     FeedForward(mlp_dim, dim, dropout),
        # )
        self.drop_path = DropPath(path_dropout) if path_dropout > 0. else nn.Identity()

    def forward_part1(self, x, y, pad_mask):
        norm_x, norm_y = self.norm1_x(x), self.norm1_y(y)  # v1 (Pre-Norm)
        x = self.drop_path(self.attn(norm_x, norm_y, norm_y, pad_mask)) + x  # v1 (Pre-Norm)
        # x = self.drop_path(self.norm1(self.attn(x, y, y, pad_mask))) + x  # v2 (Post-Norm)
        # x = self.drop_path(self.norm1(self.attn(x, y, y, pad_mask) + x))  # v3 (Add-and-Norm)
        return x

    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x))) + x  # v1 (Pre-Norm)
        # x = self.drop_path(self.norm2(self.mlp(x))) + x  # v2 (Post-Norm)
        # x = self.drop_path(self.norm2(self.mlp(x) + x))  # v3 (Add-and-Norm)
        return x

    def forward(self, x, y, pad_mask):
        B_, Nx, C = x.shape
        B_, Ny, C = y.shape
        x = self._maybe_checkpoint(self.forward_part1, x, y, pad_mask)
        x = self._maybe_checkpoint(self.forward_part2, x)
        return x
    
    def _maybe_checkpoint(self, module, *inputs):
        if self.use_checkpoint:
            return checkpoint(module, *inputs, use_reentrant = False)
        else:
            return module(*inputs)

class BasicSelfAttnLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio, dropout, path_dropout, use_checkpoint):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SelfAttnTransformerBlock(
                dim = dim,
                heads = heads[i] if isinstance(heads, list) else heads,
                dim_head = dim_head[i] if isinstance(dim_head, list) else dim_head,
                mlp_ratio = mlp_ratio,
                dropout = dropout,
                path_dropout = path_dropout[i] if isinstance(path_dropout, list) else path_dropout,
                use_checkpoint = use_checkpoint
            )
            for i in range(depth)
        ])

    def compute_selfattn_pad_mask(self, B, refer_x, pad_idx = 0, diag_unmask = False):
        if refer_x is not None:
            assert refer_x.ndim == 2
            pad_mask = compute_pad_mask(refer_x, refer_x, pad_idx, diag_unmask)
            num_repeat = B // refer_x.size(0)
            pad_mask = pad_mask.repeat_interleave(num_repeat, 0)
        else:
            pad_mask = None
        return pad_mask
    
    def forward(self, x, refer_x = None):
        B_, N, C = x.shape
        pad_mask = self.compute_selfattn_pad_mask(B_, refer_x)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return x

class BasicCrossAttnLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio, dropout, path_dropout, use_checkpoint):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            CrossAttnTransformerBlock(
                dim = dim,
                heads = heads[i] if isinstance(heads, list) else heads,
                dim_head = dim_head[i] if isinstance(dim_head, list) else dim_head,
                mlp_ratio = mlp_ratio,
                dropout = dropout,
                path_dropout = path_dropout[i] if isinstance(path_dropout, list) else path_dropout,
                use_checkpoint = use_checkpoint
            )
            for i in range(depth)
        ])
    
    def compute_crossattn_pad_mask(self, refer_x, refer_y, pad_idx = 0, diag_unmask = False):
        assert refer_x.ndim == refer_y.ndim == 2
        pad_mask = compute_pad_mask(refer_x, refer_y, pad_idx, diag_unmask)
        return pad_mask
    
    def forward(self, x, y, refer_x = None, refer_y = None):
        B_, Nx, C = x.shape
        B_, Ny, C = y.shape

        if refer_x is None:
            refer_x = torch.ones((B_, Nx))
        if refer_y is None:
            refer_y = torch.ones((B_, Ny))
        pad_mask = self.compute_crossattn_pad_mask(refer_x, refer_y)

        for blk in self.blocks:
            x = blk(x, y, pad_mask)
        return x

class STAgeViT(nn.Module):

    def __init__(self,
                 age_embed,
                 image_size, patch_size, t_length,
                 embed_dim, age_embed_dim, reduce_method,
                 spat_depth, spat_heads, spat_dim_head, spat_path_dropout,
                 temp_depth, temp_heads, temp_dim_head, temp_path_dropout,
                 age_depth, age_heads, age_dim_head, age_path_dropout,
                 temp_age_depth, temp_age_heads, temp_age_dim_head, temp_age_path_dropout,
                 spat_temp_depth, spat_temp_heads, spat_temp_dim_head, spat_temp_path_dropout,
                 mlp_ratio, dropout, embed_dropout,
                 embed_norm, encode_out_norm, decode_out_process,
                 use_checkpoint, **kwargs):
        super().__init__()
        
        self.age_embed = age_embed
        self.image_size = image_size
        self.patch_size = patch_size
        self.t_length = t_length
        self.embed_dim = embed_dim
        self.age_embed_dim = age_embed_dim
        self.reduce_method = reduce_method
        self.embed_norm = embed_norm
        self.encode_out_norm = encode_out_norm
        self.decode_out_process = decode_out_process
        self.use_checkpoint = use_checkpoint

        self.decode_patch_ratios = decode_patch_order(patch_size[-1])
        self.decode_num_layers = len(self.decode_patch_ratios)

        assert np.mod(embed_dim, patch_size[-1]) == 0
        n_groups = embed_dim // patch_size[-1]

        dps, hps, wps = patch_size
        patch_dim = dps * hps * wps
        dn, hn, wn = image_size[0] // dps, image_size[1] // hps, image_size[2] // wps

        self.patch_embedding = nn.Sequential(
            Rearrange('b t (dn dps) (hn hps) (wn wps) -> b t (dn hn wn) (dps hps wps)',
                      dps = dps, hps = hps, wps = wps, dn = dn, hn = hn, wn = wn),
            nn.Linear(patch_dim, embed_dim, bias = False),
            nn.LayerNorm(embed_dim) if embed_norm else nn.Identity(),
        )
        self.add_position_embedding = nn.Sequential(
            AddFixedSinCosPositionEmbedding(),
            nn.Dropout(embed_dropout),
        )

        spat_pdr = torch.linspace(0., spat_path_dropout, spat_depth).tolist() if spat_depth > 1 else spat_path_dropout
        self.spat_self_attn_layer = BasicSelfAttnLayer(
            dim = embed_dim,
            depth = spat_depth,
            heads = spat_heads,
            dim_head = spat_dim_head,
            mlp_ratio = mlp_ratio,
            dropout = dropout,
            path_dropout = spat_pdr,
            use_checkpoint = use_checkpoint
        )

        temp_pdr = torch.linspace(0., temp_path_dropout, temp_depth).tolist() if temp_depth > 1 else temp_path_dropout
        self.temp_self_attn_layer = BasicSelfAttnLayer(
            dim = embed_dim,
            depth = temp_depth,
            heads = temp_heads,
            dim_head = temp_dim_head,
            mlp_ratio = mlp_ratio,
            dropout = dropout,
            path_dropout = temp_pdr,
            use_checkpoint = use_checkpoint
        )

        if age_embed:
            self.age_embedding = AgeEmbedding(age_embed_dim)
            self.age_linear_embedding = nn.Sequential(
                nn.Linear(age_embed_dim, embed_dim, bias = False),
                nn.LayerNorm(embed_dim) if embed_norm else nn.Identity(),
            )
            self.age_add_position_embedding = nn.Sequential(
                Rearrange('b t d -> b t 1 d'),
                AddFixedSinCosPositionEmbedding(),
                Rearrange('b t 1 d -> b t d'),
                nn.Dropout(embed_dropout),
            )

            age_pdr = torch.linspace(0., age_path_dropout, age_depth).tolist() if age_depth > 1 else age_path_dropout
            self.age_self_attn_layer = BasicSelfAttnLayer(
                dim = embed_dim,
                depth = age_depth,
                heads = age_heads,
                dim_head = age_dim_head,
                mlp_ratio = mlp_ratio,
                dropout = dropout,
                path_dropout = age_pdr,
                use_checkpoint = use_checkpoint
            )

            temp_age_pdr = torch.linspace(0., temp_age_path_dropout, temp_age_depth).tolist() if temp_age_depth > 1 else temp_age_path_dropout
            self.temp_age_cross_attn_layer = BasicCrossAttnLayer(
                dim = embed_dim,
                depth = temp_age_depth,
                heads = temp_age_heads,
                dim_head = temp_age_dim_head,
                mlp_ratio = mlp_ratio,
                dropout = dropout,
                path_dropout = temp_age_pdr,
                use_checkpoint = use_checkpoint
            )

        spat_temp_pdr = torch.linspace(0., spat_temp_path_dropout, spat_temp_depth).tolist() if spat_temp_depth > 1 else spat_temp_path_dropout
        self.spat_temp_cross_attn_layer = BasicCrossAttnLayer(
            dim = embed_dim,
            depth = spat_temp_depth,
            heads = spat_temp_heads,
            dim_head = spat_temp_dim_head,
            mlp_ratio = mlp_ratio,
            dropout = dropout,
            path_dropout = spat_temp_pdr,
            use_checkpoint = use_checkpoint
        )

        if encode_out_norm:
            self.norm_features = nn.LayerNorm(embed_dim)
            self.norm_reduced_features = nn.LayerNorm(embed_dim)
        
        self.reverse_patch_embedding = Rearrange('b (dn hn wn) c -> b c dn hn wn', dn = dn, hn = hn, wn = wn)

        self.decode_layers = nn.ModuleList()
        in_channels = embed_dim
        
        for i_layer in range(self.decode_num_layers):
            decode_patch_r = self.decode_patch_ratios[i_layer]
            out_channels = in_channels // decode_patch_r

            decode_layer = DecodeBlock(
                in_channels = in_channels,
                mid_channels = out_channels,
                out_channels = out_channels,
                n_groups = n_groups,
                expand_r = decode_patch_r,
                dropout = dropout,
            )

            self.decode_layers.append(decode_layer)
            in_channels = out_channels

        last_decode_layer = nn.Conv3d(in_channels, 1, kernel_size = 1)
        self.decode_layers.append(last_decode_layer)
    
    def forward(self, img, age, img_mask, return_vector = False):
        output = {}
        b, t, d, h, w = img.shape
        assert ((d, h, w) == self.image_size) and (t == self.t_length)
        assert ((b, t) == age.shape) and ((d, h, w) == img_mask.shape)
        img_mask_refer = self.get_refer_from_img_mask(img_mask.unsqueeze(0).repeat(b, 1, 1, 1))
        
        img = self._maybe_checkpoint(self.patch_embedding, img)
        img = self._maybe_checkpoint(self.add_position_embedding, img)
        b, t, n, c = img.shape

        img_patches_spat = rearrange(img, 'b t n c -> (b t) n c')
        img_patches_temp = rearrange(img, 'b t n c -> (b n) t c')
        del img

        img_patches_spat = self.spat_self_attn_layer(img_patches_spat, img_mask_refer)
        img_patches_spat = self.reduce_features(img_patches_spat, age, b)  # (b, n, c)
        
        img_patches_temp = self.temp_self_attn_layer(img_patches_temp, age)
        img_patches_temp = self.reduce_features(img_patches_temp, img_mask_refer, b)  # (b, t, c)
        
        if self.age_embed:
            age_embedding = torch.stack(list(map(self.age_embedding, age)))
            age_embedding = self._maybe_checkpoint(self.age_linear_embedding, age_embedding)
            age_embedding = self._maybe_checkpoint(self.age_add_position_embedding, age_embedding)

            age_embedding = self.age_self_attn_layer(age_embedding, age)  # (b, t, c)
            img_patches_temp = self.temp_age_cross_attn_layer(img_patches_temp, age_embedding, age, age)  # (b, t, c)
        
        img_patches_spat = self.spat_temp_cross_attn_layer(img_patches_spat, img_patches_temp, img_mask_refer, age)  # (b, n, c)
        del img_patches_temp

        img_vector = self.reduce_features(img_patches_spat, img_mask_refer)  # (b, c)
        # img_vector_long = rearrange(img_patches_spat[img_mask_refer.ne(0)], '(b n) c -> b (n c)', b = b)

        if self.encode_out_norm:
            img_patches_spat = self._maybe_checkpoint(self.norm_features, img_patches_spat)
            img_vector = self._maybe_checkpoint(self.norm_reduced_features, img_vector)
        
        output = output | {'img_vector' : img_vector}

        if self.age_embed:
            img_patches_score = img_vector.softmax(-1)
            age_vector = self.reduce_features(age_embedding, age)
            age_embedding = einsum(img_patches_score, age_vector, 'b c, b c -> b')
            output = output | {'age_vector' : age_vector, 'age_cond' : age_embedding}
        
        if return_vector:
            return output
        
        img_patches_spat = self.reverse_patch_embedding(img_patches_spat)  # (b, c, dn, hn, wn)

        for decode_layer in self.decode_layers:
            img_patches_spat = self._maybe_checkpoint(decode_layer, img_patches_spat)

        if self.decode_out_process:
            img_patches_spat = self.minmax_img_cond(img_patches_spat, img_mask)

        output = output | {'img_cond' : img_patches_spat}
        del img_patches_spat
        return output

    def reduce_features(self, x, refer = None, batch_size = None):
        method = self.reduce_method

        if batch_size is not None:
            x = rearrange(x, '(b p) q c -> b p q c', b = batch_size)
            if refer is not None:
                assert refer.shape == x.shape[:2]
                replace_val = {'min' : 'inf', 'max' : '-inf', 'mean' : 'nan'}[method]
                refer = refer.ne(0)[:, :, None, None]
                x = torch.where(refer, x, float(replace_val))
        
        else:
            if refer is not None:
                assert refer.shape == x.shape[:2]
                replace_val = {'min' : 'inf', 'max' : '-inf', 'mean' : 'nan'}[method]
                refer = refer.ne(0)[:, :, None]
                x = torch.where(refer, x, float(replace_val))
        
        if method == 'min':
            x = x.amin(1)
        elif method == 'max':
            x = x.amax(1)
        elif method == 'mean':
            x = x.nanmean(1)
        else:
            raise Exception("Not supported method argument for reduce_features")
        return x
    
    def minmax_img_cond(self, img_cond, img_mask):
        img_mask = img_mask.ne(0)[None, None, :, :, :]
        mins = torch.where(img_mask, img_cond, float('inf')).amin((-3, -2, -1), keepdim = True)
        maxs = torch.where(img_mask, img_cond, float('-inf')).amax((-3, -2, -1), keepdim = True)
        img_cond = torch.where(mins < maxs, (img_cond - mins) / (maxs - mins), img_cond)
        img_cond = img_cond * img_mask
        return img_cond
    
    def get_refer_from_img_mask(self, img_mask):
        kernel = nn.MaxPool3d(self.patch_size, self.patch_size)
        refer = kernel(img_mask)
        refer = rearrange(refer, 'b dn hn wn -> b (dn hn wn)')
        return refer
    
    def _maybe_checkpoint(self, module, *inputs):
        if self.use_checkpoint:
            return checkpoint(module, *inputs, use_reentrant = False)
        else:
            return module(*inputs)



### ViT model used before - deprecated

class SelfAttentionTransformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth, heads, dim_head, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                SelfAttention(dim, heads, dim_head, dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout),
                FeedForward(mlp_dim, dim, dropout)
            ]))

    def forward(self, x, pad_mask):
        for norm1, attn, norm2, ff1, ff2 in self.layers:
            x = attn(norm1(x), pad_mask) + x
            out = ff2(ff1(norm2(x))) + x
        return out

class CrossAttentionTransformer(nn.Module):
    def __init__(self, dim, mlp_dim, heads, dim_head, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])

        self.layers.append(nn.ModuleList([
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            CrossAttention(dim, heads, dim_head, dropout),
            nn.LayerNorm(dim),
            FeedForward(dim, mlp_dim, dropout),
            FeedForward(mlp_dim, dim, dropout)
        ]))

    def forward(self, spat_x, temp_y, pad_mask):
        for norm1_1, norm1_2, attn, norm2, ff1, ff2 in self.layers:
            spat_x, temp_y = norm1_1(spat_x), norm1_2(temp_y)
            x = attn(spat_x, temp_y, temp_y, pad_mask) + spat_x
            out = ff2(ff1(norm2(x))) + x
        return out

class ViT(nn.Module):

    def __init__(self, age_embed,
                 embed_dim, age_embed_dim, scale_dim,
                 depth, temp_heads, spat_heads, cross_heads, age_heads,
                 dropout, embed_dropout, **kwargs):
        super().__init__()
        
        self.age_embed = age_embed
        self.image_size = (96, 96, 96)
        self.patch_size = (6, 6, 6)

        p1, p2, p3 = self.patch_size
        patch_dim = p1 * p2 * p3
        sn, cn, an = self.image_size[0] // p1, self.image_size[1] // p2, self.image_size[2] // p3
        self.spat_patch_num = sn * cn * an

        self.deconv_patch_ratios = decode_patch_order(self.patch_size[-1])
        self.deconv_num_layers = len(self.deconv_patch_ratios)

        assert np.mod(embed_dim, self.patch_size[-1]) == 0
        n_groups = embed_dim // self.patch_size[-1]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t (sn p1) (cn p2) (an p3) -> b t (sn cn an) (p1 p2 p3)',
                      p1 = p1, p2 = p2, p3 = p3, sn = sn, cn = cn, an = an),
            nn.Linear(patch_dim, embed_dim, bias = False),
        )

        self.add_position_embedding = nn.Sequential(
            AddFixedSinCosPositionEmbedding(),
            nn.Dropout(embed_dropout),
        )

        self.temp_reshape = Rearrange('b t n d -> (b n) t d')
        self.spat_reshape = Rearrange('b t n d -> (b t) n d')

        hidden_dim = embed_dim * scale_dim
        self.temp_transformer = SelfAttentionTransformer(embed_dim, hidden_dim, depth, temp_heads, embed_dim, dropout)
        self.spat_transformer = SelfAttentionTransformer(embed_dim, hidden_dim, depth, spat_heads, embed_dim, dropout)

        if age_embed:
            assert age_embed_dim is not None
            self.age_emb = AgeEmbedding(age_embed_dim)
            self.age_emb_expand = nn.Linear(age_embed_dim, embed_dim, bias = False)
            self.age_add_position_embedding = nn.Sequential(
                Rearrange('b t d -> b t 1 d'),
                AddFixedSinCosPositionEmbedding(),
                Rearrange('b t 1 d -> b t d')
            )
            self.age_transformer = SelfAttentionTransformer(embed_dim, hidden_dim, depth, age_heads, embed_dim, dropout)
            self.temp_age_transformer = CrossAttentionTransformer(embed_dim, hidden_dim, temp_heads, embed_dim, dropout)
        
        self.spat_temp_transformer = CrossAttentionTransformer(embed_dim, hidden_dim, cross_heads, embed_dim, dropout)
        self.from_patch_embedding = Rearrange('b (sn cn an) d -> b d sn cn an', sn = sn, cn = cn, an = an)

        img_deconv = []
        in_channels = embed_dim

        for i_layer in range(self.deconv_num_layers):
            deconv_patch_r = self.deconv_patch_ratios[i_layer]
            out_channels = in_channels // deconv_patch_r
            ksp = {2 : (4, 2, 1), 3 : (5, 3, 1)}[deconv_patch_r]
            squeeze_r = (in_channels // n_groups) // deconv_patch_r

            img_deconv_module = nn.Sequential(
                nn.GroupNorm(n_groups, in_channels),
                nn.GELU(),
                nn.ConvTranspose3d(in_channels, out_channels, *ksp),
                SEBlock(out_channels, squeeze_r),
                nn.GroupNorm(n_groups, out_channels),
                nn.GELU(),
                nn.Conv3d(out_channels, out_channels, *(3, 1, 1))
            )
            img_deconv.append(img_deconv_module)

            in_channels = out_channels
        
        self.img_deconv = nn.ModuleList(img_deconv)

        self.img_out = nn.Conv3d(out_channels, 1, kernel_size = 1)
    
    def forward(self, x, age, mask):
        assert (x.ndim == 5) and (age.ndim == 2)
        assert (mask.ndim == 3) and (x.shape[-3:] == mask.shape)
        mask = mask.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        x = self.add_position_embedding(x)

        resized_mask_flat = self.make_resized_mask_flat(mask)
        temp_pad_mask = self.make_temp_pad_mask(age)
        spat_pad_mask = self.make_spat_pad_mask(resized_mask_flat, age)
        cross_pad_mask = self.make_cross_pad_mask(resized_mask_flat, age)

        temp_x = self.temp_reshape(x)
        spat_x = self.spat_reshape(x)

        temp_x = self.temp_transformer(temp_x, temp_pad_mask)
        spat_x = self.spat_transformer(spat_x, spat_pad_mask)

        temp_x = rearrange(temp_x, '(b n) t d -> b t n d', n = n)
        spat_x = rearrange(spat_x, '(b t) n d -> b t n d', t = t)

        img_exist_mask = resized_mask_flat.ne(0)[:, None, :, None]
        temp_x_masked = torch.where(img_exist_mask, temp_x, torch.tensor(float('-inf')))
        temp_x = temp_x_masked.amax(2)
        age_exist_mask = age.ne(0)[:, :, None, None]
        spat_x_masked = torch.where(age_exist_mask, spat_x, torch.tensor(float('-inf')))
        spat_x = spat_x_masked.amax(1)

        if self.age_embed:
            age_embedding = torch.stack([self.age_emb(age[b_idx]) for b_idx in range(b)])
            age_embedding = self.age_emb_expand(age_embedding)
            age_embedding = self.age_add_position_embedding(age_embedding)
            
            age_pad_mask = self.make_age_pad_mask(age)
            age_embedding = self.age_transformer(age_embedding, age_pad_mask)
            temp_x = self.temp_age_transformer(temp_x, age_embedding, age_pad_mask)
        
        spattemp_x = self.spat_temp_transformer(spat_x, temp_x, cross_pad_mask)
        img_exist_mask = resized_mask_flat.ne(0).unsqueeze(-1)
        spattemp_x_masked = torch.where(img_exist_mask, spattemp_x, torch.tensor(float('-inf')))
        img_vector = spattemp_x_masked.amax(1)

        x = self.from_patch_embedding(spattemp_x)

        for module_idx in range(self.deconv_num_layers):
            img_deconv_module = self.img_deconv[module_idx]
            x = img_deconv_module(x)
        
        img_cond = self.img_out(x)
        mask = mask.ne(0).unsqueeze(1)
        mins = torch.where(mask, img_cond, torch.tensor(float('inf'))).amin((-3, -2, -1), keepdim = True)
        maxs = torch.where(mask, img_cond, torch.tensor(float('-inf'))).amax((-3, -2, -1), keepdim = True)
        img_cond = torch.where(mins < maxs, (img_cond - mins) / (maxs - mins), img_cond)
        img_cond *= mask

        if self.age_embed:
            spattemp_score = img_vector.softmax(-1)
            
            age_exist_mask = age.ne(0).unsqueeze(-1)
            age_embedding_masked = torch.where(age_exist_mask, age_embedding, torch.tensor(float('-inf')))
            age_embedding = age_embedding_masked.amax(1)

            age_cond = einsum(spattemp_score, age_embedding, 'b d, b d -> b')
        else:
            age_cond = torch.zeros(b, device = img_cond.device)

        out = {
            'img_cond' : img_cond,
            'age_cond' : age_cond,
            'img_vector' : img_vector
        }
        return out
    
    def make_resized_mask_flat(self, mask):
        kernel = nn.MaxPool3d(self.patch_size, self.patch_size)
        resized_mask = kernel(mask)
        resized_mask_flat = rearrange(resized_mask, 'b sn cn an -> b (sn cn an)')
        return resized_mask_flat
    
    def make_pad_mask(self, query, key, pad_idx = 0):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)
        
        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask
    
    def make_temp_pad_mask(self, age, pad_idx = 0):
        mask = self.make_pad_mask(age, age, pad_idx)
        mask = mask.repeat_interleave(self.spat_patch_num, 0)
        return mask

    def make_spat_pad_mask(self, resized_mask_flat, age, pad_idx = 0):
        mask = self.make_pad_mask(resized_mask_flat, resized_mask_flat, pad_idx)
        mask = mask.repeat_interleave(age.size(-1), 0)
        return mask

    def make_cross_pad_mask(self, resized_mask_flat, age, pad_idx = 0):
        mask = self.make_pad_mask(resized_mask_flat, age, pad_idx)
        return mask
    
    def make_age_pad_mask(self, age, pad_idx = 0):
        mask = self.make_pad_mask(age, age, pad_idx)
        return mask