import cv2
import torch
# import oneflow
import numpy as np


def tensor2uint(img):
    """convert 2/3/4-dimensional torch tensor to uint"""
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


def imread_uint(path, n_channels=3):
    """get uint8 image of size HxWxn_channles (RGB)"""
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)  # RGB
    return img


def uint2tensor4(img):
    """# convert uint to 4-dimensional torch tensor"""
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)
