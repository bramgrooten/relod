import torch
import torch.nn.functional as F


def strong_augment(obs, augm_type, overlay_alpha=0.5):
    """Augment the observation with strong augmentations."""
    if augm_type == 'conv':
        return random_conv(obs.clone())
    # elif augm_type == 'overlay':
    #   return random_overlay(obs.clone(), method='default', alpha=overlay_alpha)
    # elif augm_type == 'splice':
    #   return random_overlay(obs.clone(), method='splice')
    # elif augm_type == 'none':
    #   return obs.clone()
        # should actually skip call to heavy_augment func when heavy_aug is none
        # but since we rarely use 'none', we prefer leaving out such an if statement
    else:
        raise NotImplementedError('--augment must be one of [conv] (more to add later)')  # , overlay, splice, none]')


def random_conv(x):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.shape
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(x.device)
        temp_x = x[i:i + 1].reshape(-1, 3, h, w) / 255.
        temp_x = F.pad(temp_x, pad=[1] * 4, mode='replicate')
        out = torch.sigmoid(F.conv2d(temp_x, weights)) * 255.
        total_out = out if i == 0 else torch.cat([total_out, out], dim=0)
    return total_out.reshape(n, c, h, w)



