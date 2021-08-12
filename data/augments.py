import skimage
import torch
import torchvision
import numpy as np

from torchvision import transforms
from PIL import Image

def get_train_augmentation(opt):
    aug_list = [opt.augs]
    transf = []

    resize_aug = transforms.Resize(opt.img_size)
    crop_aug = transforms.RandomResizedCrop(opt.img_size[0], 
          scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2)

    cj = transforms.ColorJitter(0.1, 0.1, 0.1, 0)
    flip_aug = transforms.RandomHorizontalFlip()    
    norm_aug = [transforms.ToTensor(), 
        transforms.Normalize(opt.mean_norm, opt.std_norm)]
    
    if 'crop' in aug_list:
        transf.append(crop_aug)
    else:
        transf.append(resize_aug)
    if 'jitter' in aug_list:
        transf.append(cj)
    if 'flip' in aug_list:
        transf.append(flip_aug)
    
    return transforms.Compose(transf)
