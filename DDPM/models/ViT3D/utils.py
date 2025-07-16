import torch
import torch.nn as nn
import numpy as np
import math

def get_fixed_sincos_position_embedding(x_shape,
                                        temperature: float = 10000,
                                        dtype: np.dtype = np.float32):
    assert len(x_shape) in (4, 5), f'Unsupported input shape: {x_shape}'
    num_parts = 4 if len(x_shape) == 4 else 6
    channels = x_shape[-1]
    assert channels % num_parts == 0, f'Channels must be multiple of {num_parts}'
    omega = np.arange(
        channels // num_parts, dtype = np.float32) / (channels / num_parts)
    omega = 1. / (temperature**omega)

    if len(x_shape) == 4: # 2D input.
        _, h, w, _ = x_shape
        y, x = np.mgrid[:h, :w]
        y = np.einsum('m,d->md', y.flatten(), omega)
        x = np.einsum('m,d->md', x.flatten(), omega)
        p = [np.sin(x), np.cos(x), np.sin(y), np.cos(y)]
        shape = (1, h, w, channels)
    elif len(x_shape) == 5: # 3D input.
        _, t, h, w, _ = x_shape
        z, y, x = np.mgrid[:t, :h, :w]
        z = np.einsum('m,d->md', z.flatten(), omega)
        y = np.einsum('m,d->md', y.flatten(), omega)
        x = np.einsum('m,d->md', x.flatten(), omega)
        p = [np.sin(z), np.cos(z),
            np.sin(x), np.cos(x),
            np.sin(y), np.cos(y)]
        shape = (1, t, h, w, channels)
    else: # Should never reach there because of assert at beginning.
        raise ValueError(f'Unsupported input shape: {x_shape}')
    
    assert (shape[0] == 1) and (shape[1:] == x_shape[1:])
    pe = np.concatenate(p, axis=1)
    return np.asarray(pe, dtype).reshape(*shape)

class AddFixedSinCosPositionEmbedding(nn.Module):

    temperature: float = 10000
    dtype: np.dtype = np.float32

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        fixed_sincos_position_embedding = get_fixed_sincos_position_embedding(inputs.shape, self.temperature, self.dtype)
        fixed_sincos_position_embedding = torch.tensor(fixed_sincos_position_embedding, device = inputs.device)
        return inputs + fixed_sincos_position_embedding

def compute_pad_mask(query, key, pad_idx = 0, diag_unmask = False):
    _, query_seq_len = query.shape
    _, key_seq_len = key.shape
    
    key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len)
        
    mask = key_mask & query_mask  # 1, 1, query_seq_len, key_seq_len
    if diag_unmask:
        assert query_seq_len == key_seq_len
        mask[:, :, torch.arange(key_seq_len), torch.arange(key_seq_len)] = True
    mask.requires_grad = False
    return mask

def decode_patch_order(num, order = None):
    if order is None:
        order = []
    
    if num % 2 == 0:
        order.append(2)
        num = num // 2
        return decode_patch_order(num, order)
    else:
        if num % 3 == 0:
            order.append(3)
            num = num // 3
            return decode_patch_order(num, order)
        else:
            if num != 1:
                order.append(num)
            return order