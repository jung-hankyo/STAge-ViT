import torch
import torch.nn as nn
import torch.nn.functional as F

default_settings = {

    'ViT_default_settings' : {
        
        ## ViT and UNet
        'age_embed' : True,
        'age_embed_dim' : 32,
        'dropout' : 0.1,
        'use_checkpoint' : False,

        ## ViT
        'embed_dim' : 108,  # 256
        'scale_dim' : 2,
        'depth' : 1,
        'temp_heads' : 1,
        'spat_heads' : 1,  # 4
        'cross_heads' : 1,  # 4
        'age_heads' : 1,
        'embed_dropout' : 0.1,

        ## UNet
        'n_channels' : 32,
        'n_groups' : 32,
        'n_blocks' : 1,
        'ch_mults' : (2, 2, 2, 2),
        'is_attn' : (False, False, False, False),

        ## Classifier
        'use_cls_deviation' : False,
        'cls_layer_ratio' : 16,
        'cls_loss_coef' : 0.001,

        ## SimMatBuilder
        'use_sim_matrix' : False,
        'sim_type' : 'cosine',
        'sim_thres' : 0.9,
        'sim_loss_coef' : 0.01,

        ## Training params
        'lr' : 1e-4,
        'weight_decay' : 1e-4,
        'timesteps' : 1000,  # fixed to 1000
        'p_uncond' : 0.1,

    },

    'STAgeViT_default_settings' : {
        
        ## STAgeViT and UNet
        'age_embed' : True,
        'age_embed_dim' : 32,
        'dropout' : 0.0,
        'use_checkpoint' : False,

        ## STAgeViT
        'image_size' : (96, 96, 96),
        'patch_size' : (6, 6, 6),
        't_length' : 9,

        'embed_dim' : 108,
        'reduce_method' : 'max',

        'spat_depth' : 1,
        'spat_heads' : [1],
        'spat_dim_head' : [108],
        'spat_path_dropout' : 0.0,

        'temp_depth' : 1,
        'temp_heads' : [1],
        'temp_dim_head' : [108],
        'temp_path_dropout' : 0.0,

        'age_depth' : 1,
        'age_heads' : [1],
        'age_dim_head' : [108],
        'age_path_dropout' : 0.0,

        'temp_age_depth' : 1,
        'temp_age_heads' : [1],
        'temp_age_dim_head' : [108],
        'temp_age_path_dropout' : 0.0,

        'spat_temp_depth' : 1,
        'spat_temp_heads' : [1],
        'spat_temp_dim_head' : [108],
        'spat_temp_path_dropout' : 0.0,

        'mlp_ratio' : 2,
        'embed_dropout' : 0.0,

        'embed_norm' : False,
        'encode_out_norm' : False,
        'decode_out_process' : True,  # set True if img_mask provided

        ## UNet
        'n_channels' : 32,
        'n_groups' : 32,
        'n_blocks' : 1,
        'ch_mults' : (2, 2, 2, 2),
        'is_attn' : (False, False, False, False),

        ## Classifier
        'use_cls_deviation' : False,
        'cls_layer_ratio' : 1,  # deprecated
        'cls_loss_coef' : 5e-5,
        
        ## SimMatBuilder
        'use_sim_matrix' : False,
        'sim_type' : 'cosine',
        'sim_thres' : 0.9,
        'sim_loss_coef' : 0.01,

        ## Training params
        'lr' : 1e-4,
        'weight_decay' : 1e-4,
        'timesteps' : 1000,  # fixed to 1000
        'p_uncond' : 0.1,  # 0.1
        'use_vit' : True,  # whether to use transformer embedding (PLEASE set True)

    },

    'STAgeParallel_default_settings' : {
        
        ## STAgeParallel and UNet
        'age_embed' : True,
        'age_embed_dim' : 32,
        'use_checkpoint' : False,

        ## STAgeParallel
        'patch_size' : (4, 4, 4),
        'window_size' : (3, 3, 3),
        'temp_window_size' : 9,
        'spat_temp_window_size' : (27, 9),
        'in_chans' : 1,
        'embed_dim' : 32,  # 96

        'spat_depths' : [2, 2, 6, 2],
        'spat_num_heads' : [1, 2, 4, 8],  # [1, 2, 4, 8]
        'spat_head_dims' : [None, None, None, None],
        'spat_drop_path_rate' : 0.2,

        'temp_depths' : 2,
        'temp_num_heads' : 2,
        'temp_head_dims' : None,
        'temp_drop_path_rate' : 0.04,

        'spat_temp_depths' : 2,
        'spat_temp_num_heads' : 2,
        'spat_temp_head_dims' : None,
        'spat_temp_drop_path_rate' : 0.04,

        'mlp_ratio' : 4,
        'qkv_bias' : False,
        'qk_scale' : None,
        'drop_rate' : 0.1,
        'attn_drop_rate' : 0.,
        'encode_out_norm' : False,
        'decode_out_process' : True,
        'frozen_stages' : -1,

        ## UNet
        'n_channels' : 32,
        'n_groups' : 32,
        'ch_mults' : (2, 2, 2, 2),
        'is_attn' : (False, False, False, False),
        'n_blocks' : 1,
        'dropout' : 0.1,

        ## Classifier
        'use_cls_deviation' : False,
        'cls_layer_ratio' : 4,
        'cls_loss_coef' : 0.002,

        ## SimMatBuilder
        'use_sim_matrix' : False,
        'sim_thres' : -1.0,
        'sim_loss_coef' : 0.002,
        
        ## Training params
        'lr' : 1e-4,
        'weight_decay' : 1e-4,
        'timesteps' : 1000,  # fixed to 1000
        'p_uncond' : 0.1,

    },

}

timesteps = 1000

train_batch_size = 2
accumulation_steps = 2
valid_batch_size = 1
is_train_shuffle = True
is_valid_shuffle = True

loss_fn = nn.MSELoss()
cls_loss_fn = nn.CrossEntropyLoss()
sim_loss_fn = nn.L1Loss(reduction = 'sum')  # nn.MSELoss(reduction = 'sum')

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

betas = cosine_beta_schedule(timesteps)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis = 0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = '/projects1/pi/hkjung/DDPM/data/normalized_dataset3D.tar'
checkpoints_path = '/projects1/pi/hkjung/DDPM/checkpoints3D'
sample_results_path = '/projects1/pi/hkjung/DDPM/results'