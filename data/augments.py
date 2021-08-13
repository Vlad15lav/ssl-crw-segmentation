import skimage
import torch
import torchvision
import numpy as np

from torchvision import transforms
from PIL import Image

class CropPatches(object):
    def __init__(self, patch_shape=(64, 64, 3), stride=(0.5, 0.5)):
        self.patch_shape = patch_shape        
        stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
        self.stride = [int(shape[0]*stride), int(shape[1]*stride), shape[2]]

        self.spatial_jitter = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomResizedCrop(shape[0], scale=(0.7, 0.9))
        ])

    def __call__(self, x):
        if torch.is_tensor(x):
            x = x.numpy().transpose(1, 2, 0)
        elif 'PIL' in str(type(x)):
            x = np.array(x)#.transpose(2, 0, 1)
        
        winds = skimage.util.view_as_windows(x, self.patch_shape, step=self.stride)
        winds = winds.reshape(-1, *winds.shape[-3:])

        P = [transform(spatial_jitter(w)) for w in winds]
        return torch.cat(P, dim=0)

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
    transf.append(norm_aug)

    if 'grid' in aug_list:
        transf.append(CropPatches())

    return transforms.Compose(transf)
