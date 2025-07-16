import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from typing import Union, Tuple, List
from pathlib import Path

from default_configs import timesteps, alphas_cumprod, device, checkpoints_path, sample_results_path
from utils import get_mni152_mask, psnr, ssim, nrmse, define_new_models, extract, extract_seqs

def load_eval_models(model_name_at_epoch: str, **kwargs):
    load_path = os.path.join(checkpoints_path, f'{model_name_at_epoch}.tar')
    try:
        checkpoint = torch.load(load_path)
    except:
        raise Exception("No checkpoint file")
    
    model_type = checkpoint.get('model_type', 'ViT')
    settings = checkpoint['settings']
    loss_list, valid_loss_list = checkpoint['loss'], checkpoint['valid_loss']

    model_name = kwargs.get('model_name', checkpoint['model_name'])
    settings = {**settings, **kwargs.get('settings', {})}
    settings['use_checkpoint'] = False
    models_dict = define_new_models(model_type, model_name, **settings)
    settings = models_dict['settings']

    transformer = models_dict['transformer']
    if settings['use_vit']:
        transformer.load_state_dict(checkpoint.get('transformer_state_dict', checkpoint.get('vit_model_state_dict')))
        transformer = transformer.to(device)
        transformer.eval()

    unet = models_dict['unet']
    unet.load_state_dict(checkpoint.get('unet_state_dict', checkpoint.get('denoise_model_state_dict')))
    unet = unet.to(device)
    unet.eval()

    # classifier = models_dict['classifier']
    # simmatbuilder = models_dict['simmatbuilder']

    eval_models_dict = {
        
        'model_type' : model_type,
        'model_name' : model_name,
        'settings' : settings,

        'transformer' : transformer,
        'unet' : unet,
        # 'classifier' : classifier,
        # 'simmatbuilder' : simmatbuilder,

        'loss_list' : loss_list,
        'valid_loss_list' : valid_loss_list,

    }
    return eval_models_dict

@torch.no_grad()
def DDIM_sample(img_seq, age_seq, tgt_age_diff: Union[Tuple[float], List[float]],
                eval_models_dict,
                guide_w: float, eta: float, steps: int,
                seed: int = None, method = 'linear', process_xstart: bool = True,
                memory_cycle: int = 1, device = device):

    memory_cycle = steps if memory_cycle is None else memory_cycle
    assert img_seq.ndim == 5 and age_seq.ndim == 2
    img_seq, age_seq = img_seq.to(device), age_seq.to(device)

    use_vit = eval_models_dict['settings']['use_vit']
    
    transformer = eval_models_dict['transformer']
    unet = eval_models_dict['unet']
    transformer.eval() if use_vit else ...
    unet.eval()

    if method == 'linear':
        unit_length = timesteps // steps
        use_timesteps = np.asarray(list(range(0, timesteps, unit_length)))
    elif method == 'quadratic':
        use_timesteps = (np.linspace(0, np.sqrt(timesteps * 0.8), steps) ** 2).astype(int)
    else:
        raise NotImplementedError(f"sampling method {method} is not implemented")
    
    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
    
    new_betas = torch.tensor(new_betas)
    new_alphas = 1. - new_betas
    new_alphas_cumprod = torch.cumprod(new_alphas, axis = 0)
    new_alphas_cumprod_prev = F.pad(new_alphas_cumprod[:-1], (1, 0), value = 1.0)

    n_sample = img_seq.size(0)
    seed = 0 if seed is None else seed
    seed_generator = torch.Generator(device = device)

    x_s = torch.randn((n_sample, 1, *img_seq.shape[-3:]), generator = seed_generator.manual_seed(seed), device = device)
    template_mask = torch.tensor(get_mni152_mask(), device = device)
    
    if use_vit:
        c = transformer(img_seq, age_seq, template_mask)
    else:  # STAge-ViT-ablated Last-IE (image embedding) model implementation
        tgt_age_diff = [0.] * n_sample
        c = {'img_cond' : img_seq[torch.arange(n_sample), age_seq.ne(0).sum(-1) - 1].unsqueeze(1).to(device),
             'age_cond' : torch.zeros((n_sample,), device = device)}
    
    cond_uncond = {}
    for c_key, c_value in c.items():
        z = torch.zeros_like(c_value, dtype = torch.float, device = device)
        cond_uncond[c_key] = torch.cat((c_value, z), 0)

    tgt_age_diff = torch.tensor(tgt_age_diff, dtype = torch.float, device = device)
    z = torch.zeros_like(tgt_age_diff, dtype = torch.float, device = device)
    tgt_age_diff = torch.cat((tgt_age_diff, z), 0)
    
    one_mask = torch.tensor([1.] * (n_sample * 2), device = device)
    template_mask = template_mask[None, None, :, :, :]
    
    sampled_images, sampled_steps = [], []

    for step in tqdm(reversed(range(0, steps)), desc = f'(w: {guide_w}) (eta: {eta})', total = steps):

        x_s *= template_mask
        x_ss = torch.cat((x_s, x_s), 0)
        curr_t = torch.full((n_sample,), use_timesteps[step], dtype = torch.long, device = device).repeat(2)
        
        output_cond_uncond = unet(x_ss, tgt_age_diff, cond_uncond, one_mask, curr_t)
        output_cond = output_cond_uncond[:n_sample]
        output_uncond = output_cond_uncond[n_sample:]
        
        # v_theta_t = (1 + guide_w) * output_cond - guide_w * output_uncond  # deprecated
        v_theta_t = guide_w * output_cond + (1 - guide_w) * output_uncond
        v_theta_t *= template_mask
        
        step_t = torch.tensor([step], dtype = torch.long, device = device)
        alphas_cumprod_t = extract(new_alphas_cumprod, step_t, x_s.shape)
        alphas_cumprod_prev_t = extract(new_alphas_cumprod_prev, step_t, x_s.shape)

        alpha_t, sigma_t = torch.sqrt(alphas_cumprod_t), torch.sqrt(1. - alphas_cumprod_t)

        x0_pred = x_s * alpha_t - v_theta_t * sigma_t
        if process_xstart:
            x0_pred = torch.clamp(x0_pred, 0, 1)
            # below process_xstart method deprecated (found unnecessary and worse)
            """
            if step == 0:  # only at the last step (this changes distribution!)
                mins = torch.where(template_mask.ne(0), x0_pred, torch.tensor(float('inf'))).amin((-3, -2, -1), keepdim = True)
                maxs = torch.where(template_mask.ne(0), x0_pred, torch.tensor(float('-inf'))).amax((-3, -2, -1), keepdim = True)
                x0_pred = torch.where(mins < maxs, (x0_pred - mins) / (maxs - mins), x0_pred)
                x0_pred *= template_mask
            """
        eps_pred = x_s * sigma_t + v_theta_t * alpha_t

        ddim_sigma = (
            eta * \
            torch.sqrt((1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t)) * \
            torch.sqrt(1. - alphas_cumprod_t / alphas_cumprod_prev_t)
        )
        adjusted_sigma = torch.sqrt(1. - alphas_cumprod_prev_t - ddim_sigma ** 2)
        mean_pred = x0_pred * torch.sqrt(alphas_cumprod_prev_t) + eps_pred * adjusted_sigma
        
        if step == 0:
            x_s = mean_pred.type(torch.float)  # == x0_pred (mathematically)
        else:
            noise_seed = steps * seed + step
            noise = torch.randn(*x_s.shape[-3:], generator = seed_generator.manual_seed(noise_seed), device = device)
            noise = noise[None, None, :, :, :]
            x_s = (mean_pred + ddim_sigma * noise).type(torch.float)

        if step % memory_cycle == 0:
            sampled_images.append(x_s.detach().cpu().numpy())
            sampled_steps.append(steps - step)
    
    sampled_images, sampled_steps = np.array(sampled_images), np.array(sampled_steps)
    return sampled_images, sampled_steps
    
def save_DDIM_sample(eval_models_dict,
                     data_ID: int,
                     dataset,
                     guide_w_list: List[float],
                     eta_list: List[float],
                     steps_list: List[int],
                     seed_avg: int,
                     folder_suffix: str = 'samples',
                     **kwargs):
    
    model_name_at_epoch = eval_models_dict['model_name'] + '_e' + str(eval_models_dict['loss_list'][-1][0])
    results_path = os.path.join(sample_results_path, f'{model_name_at_epoch}_{folder_suffix}')
    path = Path(results_path)
    path.mkdir(exist_ok = True)
    
    data = dataset[data_ID]
    img_seq, age_seq = data['image'].unsqueeze(0), data['age'].unsqueeze(0)
    extracted_seqs = extract_seqs(img_seq, age_seq, is_last = True, use_age_diff = False)

    cond_img_seq, cond_age_seq = extracted_seqs['cond_img_seq'], extracted_seqs['cond_age_seq']
    tgt_img, tgt_age_diff = extracted_seqs['tgt_img_seq'], extracted_seqs['tgt_age_diff_seq']
    tgt_img = tgt_img.squeeze(0).numpy()
    tgt_age_diff = tgt_age_diff.tolist()

    for guide_w in guide_w_list:
        for eta in eta_list:
            for steps in steps_list:
                last_sampled_img_list = []
                for seed in range(0, seed_avg): #
                    sampled_imgs, _ = DDIM_sample(cond_img_seq, cond_age_seq, tgt_age_diff,
                                                  eval_models_dict,
                                                  guide_w, eta, steps,
                                                  seed = seed, memory_cycle = None,
                                                  **kwargs)
                    last_sampled_img_list.append(sampled_imgs[-1, 0, 0])
                last_sampled_img = np.stack(last_sampled_img_list, 0).mean(0)

                psnr_with_tgt = psnr(tgt_img, last_sampled_img, data_range = 1)
                ssim_with_tgt = ssim(tgt_img, last_sampled_img, data_range = 1)
                nrmse_with_tgt = nrmse(tgt_img, last_sampled_img)

                save_dict = {

                    'last_sampled_img' : last_sampled_img,
                    'psnr_with_tgt' : psnr_with_tgt,
                    'ssim_with_tgt' : ssim_with_tgt,
                    'nrmse_with_tgt' : nrmse_with_tgt

                }
                save_path = os.path.join(results_path, f'{data_ID}_{guide_w}_{eta}_{steps}.tar') #
                torch.save(save_dict, save_path)

def load_DDIM_sample(model_name_at_epoch: str,
                     data_ID: int,
                     guide_w_list: List[float],
                     eta_list: List[float],
                     steps_list: List[int],
                     folder_suffix: str = 'samples'):
    
    last_sampled_img_set = []
    psnr_with_tgt_set = []
    ssim_with_tgt_set = []
    nrmse_with_tgt_set = []

    for guide_w in guide_w_list:
        for eta in eta_list:
            for steps in steps_list:
                load_path = os.path.join(sample_results_path, '{}_{}/{}_{}_{}_{}.tar'.format(
                    model_name_at_epoch, folder_suffix, data_ID, guide_w, eta, steps
                ))

                loaded_dict = torch.load(load_path)
                last_sampled_img = loaded_dict['last_sampled_img']
                psnr_with_tgt = loaded_dict['psnr_with_tgt']
                ssim_with_tgt = loaded_dict['ssim_with_tgt']
                nrmse_with_tgt = loaded_dict['nrmse_with_tgt']

                last_sampled_img_set.append(last_sampled_img)
                psnr_with_tgt_set.append(psnr_with_tgt)
                ssim_with_tgt_set.append(ssim_with_tgt)
                nrmse_with_tgt_set.append(nrmse_with_tgt)
    
    return_dict = {

        'last_sampled_img_set' : np.array(last_sampled_img_set),
        'psnr_with_tgt_set' : np.array(psnr_with_tgt_set),
        'ssim_with_tgt_set' : np.array(ssim_with_tgt_set),
        'nrmse_with_tgt_set' : np.array(nrmse_with_tgt_set)

    }
    return return_dict