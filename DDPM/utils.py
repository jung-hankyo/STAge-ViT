import torch
from torch.optim import Adam, AdamW, lr_scheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
from typing import Tuple, List
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from prettytable import PrettyTable
import SimpleITK as sitk
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
import ants

from default_configs import *

def define_new_models(model_type: str, model_name: str, **kwargs):

    # import backbone model to define [condition extraction]
    if model_type == 'ViT':
        local_default_settings = default_settings['ViT_default_settings']
        from models.ViT3D.ViT import ViT as Transformer
    elif model_type == 'STAgeViT':
        local_default_settings = default_settings['STAgeViT_default_settings']
        from models.ViT3D.ViT import STAgeViT as Transformer
    elif model_type == 'STAgeParallel':
        local_default_settings = default_settings['STAgeParallel_default_settings']
        from models.SwinT3D.SwinT3D import STAgeParallelTransformer as Transformer
    else:
        raise Exception("Not supported backbone model")
    
    # import UNet model [image generation]
    from models.UNet3D.UNet import UNet

    # make settings based on default_settings and kwargs
    settings = {**local_default_settings, **kwargs}
    assert settings['timesteps'] == 1000, "Timesteps value is fixed to 1000"

    # define basic models and register parameters
    unet = UNet(**settings).to(device)
    params = list(unet.parameters())
    if settings['use_vit']:
        transformer = Transformer(**settings).to(device)
        params += list(transformer.parameters())
    else:
        transformer = None

    # import accessory modules; these modules are related to loss computation for training
    if settings['use_cls_deviation']:
        from models.Accessories.modules import ClassDeviationClassifier as Classifier
        if model_type in ('ViT', 'STAgeViT'):
            dim0 = settings['embed_dim']
            dim1 = int(dim0 / settings['cls_layer_ratio'])
            drop = settings['dropout']
        elif model_type == 'STAgeParallel':
            dim0 = int(settings['embed_dim'] * 2 ** (len(settings['spat_depths']) - 1))
            dim1 = int(dim0 / settings['cls_layer_ratio'])
            drop = settings['drop_rate']
        classifier = Classifier(dim0, dim1, drop).to(device)
        params += list(classifier.parameters())
    else:
        classifier = None
    
    if settings['use_sim_matrix']:
        from models.Accessories.modules import SimilarityMatrixBuilder as SimMatBuilder
        simmatbuilder = SimMatBuilder(**settings).to(device)
        # params += list(simmatbuilder.parameters()) # zero
    else:
        simmatbuilder = None
    
    # define optimizer and scheduler
    # optimizer = Adam(params, lr = settings['lr'], weight_decay = settings['weight_decay'])
    optimizer = AdamW(params, lr = settings['lr'], weight_decay = settings['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.99)

    models_dict = {

        'model_type' : model_type,
        'model_name' : model_name,
        'settings' : settings,

        'transformer' : transformer,
        'unet' : unet,
        'classifier' : classifier,
        'simmatbuilder' : simmatbuilder,

        'optimizer' : optimizer,
        'scheduler' : scheduler,

        'loss_list' : [],
        'valid_loss_list' : [],

    }
    return models_dict

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def image_padding(raw_image, frame_num = 9):
    time_num = raw_image.shape[0]
    assert time_num in np.arange(frame_num) + 1, "Not intended frame number of input subject."
    
    padding_num = frame_num - time_num
    if padding_num == 0:
        padded_img = raw_image
    else:
        padding = torch.zeros((padding_num, *raw_image.shape[1:]), dtype = torch.float)
        padded_img = torch.cat((raw_image, padding), dim = 0)
    return padded_img

def age_padding(raw_age, frame_num = 9):
    length = len(raw_age)
    assert length in np.arange(frame_num) + 1, "Not intended frame number of input subject."
    
    padding_num = frame_num - length
    if padding_num == 0:
        padded_age = raw_age
    else:
        padding = torch.zeros(padding_num, dtype = torch.float)
        padded_age = torch.cat((raw_age, padding), dim = -1)
    return padded_age

def show_images(image, variety = False, figsize = (6, 2)): # default_dim = '3D'
    assert image.ndim == 3
    if type(image) == torch.Tensor:
        image = image.numpy()
    rotate_ccw = lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if variety == False:
        x, y, z = image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize = figsize)
        axes[0].imshow(rotate_ccw(image[x, :, :]), 'jet')
        axes[1].imshow(rotate_ccw(image[:, y, :]), 'jet')
        axes[2].imshow(rotate_ccw(image[:, :, z]), 'jet')
        for ax in axes:
            ax.set_axis_off()
        plt.tight_layout()
    
    else:
        x = np.linspace(0, image.shape[0], 12).astype(np.int64)[1:-1]
        y = np.linspace(0, image.shape[1], 12).astype(np.int64)[1:-1]
        z = np.linspace(0, image.shape[2], 12).astype(np.int64)[1:-1]

        fig, axes = plt.subplots(1, 10, figsize = figsize)
        for j in range(10):
            axes[j].imshow(rotate_ccw(image[x[j], :, :]), 'jet')
            axes[j].set_axis_off()
        plt.tight_layout()
        plt.show()
        
        fig, axes = plt.subplots(1, 10, figsize = (10, 1))
        for j in range(10):
            axes[j].imshow(rotate_ccw(image[:, y[j], :]), 'jet')
            axes[j].set_axis_off()
        plt.tight_layout()
        plt.show()
        
        fig, axes = plt.subplots(1, 10, figsize = (10, 1))
        for j in range(10):
            axes[j].imshow(rotate_ccw(image[:, :, z[j]]), 'jet')
            axes[j].set_axis_off()
        plt.tight_layout()
        plt.show()

def get_img_info(nib_img):
    affine_matrix = nib_img.affine
    spacing = np.array(nib_img.header.get_zooms(), dtype = np.float64)
    origin = affine_matrix[:3, 3]
    #direction = affine_matrix[:3, :3].flatten()
    return (spacing, origin)

def convert_to_sitk(img_volume, spacing, origin, direction = None):
    """convert numpy volume to sitk image"""
    img_volume = img_volume.transpose(2, 1, 0) # since sitk assume numpy in [z, y, x]
    sitk_volume = sitk.GetImageFromArray(img_volume.astype(np.float64))
    sitk_volume.SetSpacing(spacing)
    sitk_volume.SetOrigin(origin)
    if direction:
        sitk_volume.SetDirection(direction)
    return sitk_volume

def convert_to_numpy(sitk_volume):
    """convert sitk image to numpy volume"""
    img_volume = sitk.GetArrayFromImage(sitk_volume)
    img_volume = img_volume.transpose(2, 1, 0)
    return img_volume

def resample(sitk_volume, new_spacing, new_size, default_value = 0.):
    """1) Create resampler"""
    resample = sitk.ResampleImageFilter()
    
    """2) Set parameters"""
    #set interpolation method, output direction, default pixel value
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(sitk_volume.GetDirection())
    resample.SetDefaultPixelValue(default_value)
    
    #set output spacing
    new_spacing = np.array(new_spacing)
    resample.SetOutputSpacing(new_spacing)
    
    #set output size and origin
    old_size = np.array(sitk_volume.GetSize())
    old_spacing = np.array(sitk_volume.GetSpacing())
    new_size_no_shift = np.int16(np.ceil(old_size*old_spacing/new_spacing))
    old_origin = np.array(sitk_volume.GetOrigin())
    
    shift_amount = np.int16(np.floor((new_size_no_shift - new_size)/2))*new_spacing
    new_origin = old_origin + shift_amount
    
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetOutputOrigin(new_origin)
    
    """3) execute"""
    new_volume = resample.Execute(sitk_volume)
    return new_volume

def resample_pet_image(pet_nifti_img):
    """Resample an original 3D PET image of resolution 1.5mm3 and size (160, 160, 96)
       into a deformed image of resolution 2.0mm3 and size (96, 96, 96)"""
    spacing, origin = get_img_info(pet_nifti_img)
    img = convert_to_sitk(pet_nifti_img.get_fdata(), spacing, origin)
    img = resample(img, (2.0, 2.0, 2.0), (96, 96, 96))
    img = convert_to_numpy(img)
    return img

def get_mni152_template():
    """Load MNI152 template image of resolution 1.0mm3 and size (197, 233, 189)
       and resample it into resolution 2.0mm3 and size (96, 96, 96)"""
    mni_template = load_mni152_template(resolution = 1.0)
    spacing, origin = get_img_info(mni_template)
    mni_template = convert_to_sitk(mni_template.get_fdata(), spacing, origin)
    mni_template = resample(mni_template, (2.0, 2.0, 2.0), (96, 96, 96))
    mni_template = convert_to_numpy(mni_template)
    return mni_template

def get_mni152_mask(threshold = 0.2):
    """Load MNI152 brain mask of resolution 1.0mm3 and size (197, 233, 189)
       and resample it into resolution 2.0mm3 and size (96, 96, 96)"""
    mni_mask = load_mni152_brain_mask(resolution = 1.0, threshold = threshold)
    spacing, origin = get_img_info(mni_mask)
    mni_mask = convert_to_sitk(mni_mask.get_fdata(), spacing, origin)
    mni_mask = resample(mni_mask, (2.0, 2.0, 2.0), (96, 96, 96))
    mni_mask = convert_to_numpy(mni_mask)
    assert (np.unique(mni_mask) == (0., 1.)).all() # assert binary mask
    return mni_mask

def get_register_matrix(pet_image, template_image):
    """Get registration (transform) matrix under SyN (symmetric normalization) method"""
    assert pet_image.shape == template_image.shape
    pet_image = ants.from_numpy(pet_image)
    template_image = ants.from_numpy(template_image)
    reg = ants.registration(fixed = template_image, moving = pet_image, type_of_transform = 'SyN')
    return reg['fwdtransforms']

def register_pet_to_template(pet_image, template_image, transform_matrix, mask = None):
    """Do registration with pre-obtained registration (transform) matrix"""
    assert pet_image.shape == template_image.shape
    pet_image = ants.from_numpy(pet_image)
    template_image = ants.from_numpy(template_image)
    registered_pet_image = ants.apply_transforms(fixed = template_image, moving = pet_image, transformlist = transform_matrix)
    registered_pet_image = ants.core.ants_image.ANTsImage.numpy(registered_pet_image)
    if mask is not None:
        assert pet_image.shape == mask.shape
        registered_pet_image *= mask
        registered_pet_image[mask == 1] -= registered_pet_image[mask == 1].min()
        registered_pet_image[mask == 1] /= registered_pet_image[mask == 1].max()
    return registered_pet_image

def canny_masking(image, threshold1 = 55, threshold2 = 110):
    if type(image) == torch.Tensor:
        image = image.numpy()
    
    assert image.ndim == 2
    assert (image.min(), image.max()) == (0.0, 1.0)
    
    if type(image) == torch.Tensor:
        image = image.numpy()
    
    # pre-preprocess (roughly remove skull signal and strike artifact)
    n, b = np.histogram(image, bins = 200, range = (0.0, 1.0))
    
    peaks, _ = find_peaks(n, height = (0, 1000), distance = 50)
    image = np.clip(image, (b[peaks[0]] + 0.0025) * 1.5, 1.0)
    image = (image - image.min()) / (image.max() - image.min())

    # preprocess
    image = (image * 255).astype(np.uint8)
    edges = cv2.Canny(image, threshold1, threshold2, apertureSize = 3, L2gradient = True)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations = 2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations = 3)
    
    inv_new_edges = cv2.bitwise_not(edges)
    h, w = inv_new_edges.shape
    mask_ = np.zeros((h + 2, w + 2), np.uint8)
    
    _ = cv2.floodFill(inv_new_edges, mask_, (0, 0), 0)

    mask = cv2.bitwise_or(edges, inv_new_edges)
    label_num, _ = cv2.connectedComponents(cv2.bitwise_not(mask))
    assert label_num == 2

    mask = (mask // 255).astype(np.float32)
    mask = cv2.medianBlur(mask, 5)
    return mask

def make_common_mask(img_seq, age_seq, smooth_mask = True, **kwargs):
    assert (img_seq.ndim == 4) and (age_seq.ndim == 1)

    last_idx = int(age_seq.nonzero()[-1].item())
    common_mask = np.ones(img_seq.shape[-2:], dtype = np.float32)

    for i in range(last_idx + 1):
        mask = canny_masking(img_seq[i, 0], **kwargs)
        common_mask = cv2.bitwise_and(common_mask, mask)
    
    if smooth_mask:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        common_mask = cv2.morphologyEx(common_mask, cv2.MORPH_OPEN, kernel, iterations = 3)
        common_mask = cv2.morphologyEx(common_mask, cv2.MORPH_DILATE, kernel, iterations = 1)
        common_mask = cv2.morphologyEx(common_mask, cv2.MORPH_CLOSE, kernel, iterations = 3)
        common_mask = cv2.medianBlur(common_mask, 5)
    
    return common_mask

"""
def remove_strike_artifact(image, hist_bins = 1000, find_range = 50, method = 'aggressive'):
    if type(image) == torch.Tensor:
        image = image.numpy()
    
    assert image.ndim == 2
    assert (image.min(), image.max()) == (0.0, 1.0)

    n, b = np.histogram(image.reshape(-1), bins = hist_bins, range = (0.0, 1.0))

    global_max_idx = n.argmax(0)
    find_start_idx = global_max_idx + 1
    truncated_arr = n[find_start_idx:]

    for idx, value in enumerate(b[find_start_idx:]):
        find_arr = truncated_arr[idx:(idx + find_range + 1)]
        if find_arr.argmin(0) == 0:
            local_min_idx = idx + find_start_idx
            break

    if method == 'conservative': # preserve skull
        thres_value = b[local_min_idx + 1]
        _, mask = cv2.threshold(image, thresh = thres_value, maxval = 1.0, type = cv2.THRESH_BINARY)
    
    elif method == 'aggressive': # remove skull, preserve brain parenchyma
        second_global_max_idx = n[local_min_idx:].argmax(0) + local_min_idx

        if b[second_global_max_idx] > 0.3: # something wrong so replace it
            second_local_min_idx = local_min_idx
        else: # okay
            find_start_idx = second_global_max_idx + 1
            truncated_arr = n[find_start_idx:]
            
            for idx, value in enumerate(b[find_start_idx:]):
                find_arr = truncated_arr[idx:(idx + find_range + 1)]
                if find_arr.argmin(0) == 0:
                    second_local_min_idx = idx + find_start_idx
                    break
        
        thres_value = b[second_local_min_idx + 1]
        _, mask = cv2.threshold(image, thresh = thres_value, maxval = 1.0, type = cv2.THRESH_BINARY)
    
    else:
        raise Exception("Not supported method. Only 'conservative' or 'aggressive' method supported.")
    print(global_max_idx, local_min_idx, second_global_max_idx, second_local_min_idx)
    return mask

def make_common_mask(img_seq, age_seq, morph_transform = True, **kwargs):
    assert (img_seq.ndim == 4) and (age_seq.ndim == 1)

    last_idx = int(age_seq.nonzero()[-1].item())
    common_mask = np.ones(img_seq.shape[-2:], dtype = np.float32)

    for i in range(last_idx + 1):
        mask = remove_strike_artifact(img_seq[i, 0], **kwargs)
        common_mask = cv2.bitwise_and(common_mask, mask)
    
    common_mask = common_mask.astype(np.uint8)
    inv_common_mask = 1 - common_mask
    label_num, _ = cv2.connectedComponents(inv_common_mask)

    assert label_num >= 2

    if label_num == 2:
        filled_mask = common_mask
    else:
        h, w = inv_common_mask.shape
        mask_ = np.zeros((h + 2, w + 2), np.uint8)

        _ = cv2.floodFill(inv_common_mask, mask_, (0, 0), 0)

        filled_mask = cv2.bitwise_or(common_mask, inv_common_mask)
        label_num, _ = cv2.connectedComponents(1 - filled_mask)
        assert label_num == 2
    
    if morph_transform:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel, iterations = 10)
    
    return filled_mask
"""

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise = None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_v_target(x_start, t, noise):
    return (
        extract(sqrt_alphas_cumprod, t, x_start.shape) * noise - \
        extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
    )

def extract_seqs(img_seq, age_seq, is_last = True, use_age_diff = True):
    assert (img_seq.ndim == 5) and (age_seq.ndim == 2)

    batch_size = img_seq.shape[0]
    batch_indices = torch.arange(batch_size)

    last_indices = age_seq.ne(0).sum(-1) - 1

    if is_last:
        tgt_indices = last_indices
    else:
        tgt_indices = [torch.randint(1, last_idx + 1, (1,)).item() for last_idx in last_indices]
    
    tgt_img_seq = img_seq[batch_indices, tgt_indices]
    tgt_age_seq = age_seq[batch_indices, tgt_indices]

    cond_img_seq = img_seq.clone()
    cond_age_seq = age_seq.clone()
    cond_img_seq[batch_indices, tgt_indices] *= 0
    cond_age_seq[batch_indices, tgt_indices] *= 0

    cond_last_indices = tgt_indices - 1
    cond_last_img_seq = cond_img_seq[batch_indices, cond_last_indices]

    if use_age_diff:
        cond_last_age_seq = cond_age_seq[batch_indices, cond_last_indices]
        tgt_age_diff_seq = tgt_age_seq - cond_last_age_seq
    else:
        tgt_age_diff_seq = tgt_age_seq

    extracted_seqs = {

        'cond_last_img_seq' : cond_last_img_seq,
        'cond_img_seq' : cond_img_seq,
        'cond_age_seq' : cond_age_seq,
        'tgt_img_seq' : tgt_img_seq,
        'tgt_age_diff_seq' : tgt_age_diff_seq

    }
    return extracted_seqs

def analyze_loss(loss_list):
    loss_columns = range(1, len(loss_list[0]) - 1)
    df = pd.DataFrame(loss_list, columns = ['epoch', 'batch', *loss_columns])
    dfs = df.groupby(by = 'epoch')
    df = pd.DataFrame(map(lambda df: df[1].mean(0), dfs))

    df['epoch'] = pd.array(df['epoch'], dtype = int)
    df = df.drop(columns = ['batch'])
    df['tot_avg_loss'] = df.iloc[:, 1:].sum(1)
    return df

def draw_loss_plot(*loss_lists, **plot_kwargs):

    plot_settings = {

        'linewidths' : 1.0,
        'linestyles' : '-',
        'colors' : ['r', 'b', 'g', 'y', 'm', 'c', 'k'],
        'alphas' : 0.7,
        'labels' : None,
        'xlabel' : 'epoch',

    }

    plot_settings = {**plot_settings, **plot_kwargs}

    plt.figure(figsize = plot_settings.get('figsize'))

    for i, loss_list in enumerate(loss_lists):

        set_value = lambda param: param[i] if isinstance(param, (List, Tuple)) else param

        x = np.arange(len(loss_list)) + 1
        plt.plot(x, loss_list,
                 linewidth = set_value(plot_settings.get('linewidths')),
                 linestyle = set_value(plot_settings.get('linestyles')),
                 color = set_value(plot_settings.get('colors')),
                 alpha = set_value(plot_settings.get('alphas')),
                 label = set_value(plot_settings.get('labels')))
    
    plt.semilogy(base = 10)
    plt.title('Loss by {}'.format(plot_settings.get('xlabel').capitalize()))
    plt.xlabel(plot_settings.get('xlabel'))
    plt.ylabel('Loss (log)')
    plt.legend() if plot_settings.get('labels') else ...
    plt.show()

def compare_loss_plot(*model_names, by_epoch = True, loss_type = 1, **kwargs):
    losses = []
    for model_name in model_names:
        loss = torch.load('./checkpoints3D/' + model_name + '.tar')['loss']
        if by_epoch:
            loss = analyze_loss(loss).iloc[:, loss_type]
        else:
            loss = pd.DataFrame(loss).drop(columns = 1).iloc[:, loss_type]
        losses.append(loss)
    if by_epoch:
        kwargs = {'alphas' : 0.8, 'linewidths' : 0.9, 'labels' : model_names, 'xlabel' : 'epoch'} | kwargs
    else:
        kwargs = {'alphas' : 0.4, 'linewidths' : 0.5, 'labels' : model_names, 'xlabel' : 'step'} | kwargs
    draw_loss_plot(*losses, **kwargs)

"""
def show_sample_process(sampled_images, sampled_steps, sampled_idx): # default_dim = '2D'
    assert sampled_images.ndim == 5 and sampled_steps.ndim == 1
    show_images = sampled_images[:, sampled_idx, 0]
    show_steps = sampled_steps

    ncols = 10
    nrows = len(show_steps) // ncols + (len(show_steps) % ncols != 0)
    fig = plt.figure(figsize = (ncols * 1.5, nrows * 1.5))
    
    for i in range(len(show_steps)):
        fig.add_subplot(nrows, ncols, i + 1)
        show_image = show_images[i]
        plt.imshow(show_image, 'jet')
        plt.title(f'{show_steps[i]}-step', fontsize = 8.)
        plt.axis('off')
    plt.show()
"""

def nrmse(p, q):
    assert p.shape == q.shape, "p and q must have same shape"

    deviation = p - q
    mse = np.mean(np.square(deviation))
    rmse = np.sqrt(mse)
    return rmse / (p.max() - p.min())

def quantify_sample_process(pred_images,
                            target_image,
                            print_value = True,
                            show_graph = True,
                            show_image = False,
                            show_hist = False,
                            lookup_idx = -1):
    
    assert pred_images.shape[1:] == (1, 1, *target_image.shape)
    if type(target_image) == torch.Tensor:
        target_image = target_image.numpy()

    pred_images = pred_images.reshape(-1, *target_image.shape)
    
    p = [psnr(target_image, pred_images[i], data_range = 1) for i in range(len(pred_images))]
    s = [ssim(target_image, pred_images[i], data_range = 1) for i in range(len(pred_images))]
    n = [nrmse(target_image, pred_images[i]) for i in range(len(pred_images))]

    pi, si, ni = np.argmax(p), np.argmax(s), np.argmin(n)
    
    if print_value:
        print(f'Optimal (PSNR, SSIM, NRMSE) = ({p[pi]:>2f}, {s[si]:>2f}, {n[ni]:>2f}) at ({(pi+1)}, {(si+1)}, {(ni+1)})')
        print(f'Last (PSNR, SSIM, NRMSE) = ({p[-1]:>2f}, {s[-1]:>2f}, {n[-1]:>2f}) at ({len(pred_images)})')
    
    if show_graph:
        _, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 4))
        x = np.arange(len(pred_images)) + 1
        axes[0].set_title("PSNR", fontsize = 12.); axes[0].set_xlabel("step"); axes[0].plot(x, p, 'b')
        axes[1].set_title("SSIM", fontsize = 12.); axes[1].set_xlabel("step"); axes[1].plot(x, s, 'b')
        axes[2].set_title("NRMSE", fontsize = 12.); axes[2].set_xlabel("step"); axes[2].plot(x, n, 'b')
        plt.tight_layout()
        plt.show()
    
    if show_image:
        lookup_pred_image = pred_images[lookup_idx]
        show_images(lookup_pred_image)
        show_images(target_image)
        show_images(lookup_pred_image - target_image)
    
    if show_hist:
        lookup_pred_image = pred_images[lookup_idx]
        plt.hist(lookup_pred_image.reshape(-1), bins = 512, color = 'b', alpha = 0.3, label = "Pred")
        plt.hist(target_image.reshape(-1), bins = 512, color = 'r', alpha = 0.3, label = "Target")
        plt.legend()
        plt.show()