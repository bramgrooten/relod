import os
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as TF

# global variables for places365 dataset
places_dataloader = None
places_iter = None


def strong_augment(obs, augm_type, overlay_alpha=0.5):
    """Augment the observation with strong augmentations."""
    if augm_type == 'conv':
        return random_conv(obs.clone())
    elif augm_type == 'overlay':
      return random_overlay(obs.clone(), alpha=overlay_alpha)
    # elif augm_type == 'splice':
    #   return random_overlay(obs.clone(), method='splice')
    else:
        raise NotImplementedError('--augment must be one of [conv, overlay]')


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


def random_overlay(x, dataset='places365_standard', alpha=0.5):
    """Randomly overlay an image from Places"""
    global places_iter
    if dataset == 'places365_standard':
        if places_dataloader is None:
            _load_places(batch_size=x.size(0), image_size=(x.size(-2), x.size(-1)))
        imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1) * 255.0
    else:
        raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')
    return imgs * alpha + x * (1 - alpha)


def _load_places(batch_size=256, image_size=(90, 160), num_workers=8, use_val=False):
    global places_dataloader, places_iter

    data_dir = os.environ.get('DMCGB_DATASETS')
    assert data_dir is not None, 'DMCGB_DATASETS not set. Use `export DMCGB_DATASETS="/path/to/datasets"`'
    print('DMCGB_DATASETS:', os.environ['DMCGB_DATASETS'])
    print('data_dir', data_dir)

    partition = 'val' if use_val else 'train'
    print(f'Loading {partition} partition of places365_standard...')

    if os.path.exists(data_dir):
        fp = os.path.join(data_dir, 'places365_standard', partition)
        if not os.path.exists(fp):
            print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
            fp = data_dir
        places_dataloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(fp, TF.Compose([
                TF.RandomResizedCrop(image_size),
                TF.RandomHorizontalFlip(),
                TF.ToTensor()
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        places_iter = iter(places_dataloader)
        # print('Loaded image augmentation dataset from', data_dir)
    else:
        raise FileNotFoundError(f'Failed to find places365 data at {data_dir}')

    if places_iter is None:
        raise FileNotFoundError(f'Failed to load places365 data at {data_dir}')


def _get_places_batch(batch_size):
    global places_iter
    try:
        imgs, _ = next(places_iter)
        if imgs.size(0) < batch_size:
            places_iter = iter(places_dataloader)
            imgs, _ = next(places_iter)
    except StopIteration:
        places_iter = iter(places_dataloader)
        imgs, _ = next(places_iter)
    return imgs.cuda()
