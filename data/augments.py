import skimage
import torch
import torchvision
import numpy as np

from torchvision import transforms
from PIL import Image

class CropPatches(object):
    def __init__(self, transform, patch_shape=(64, 64, 3), stride=(0.5, 0.5)):
        self.transform = transforms.Compose(transform)
        self.patch_shape = patch_shape
        
        stride = np.random.random() * (stride[1] - stride[0]) + stride[0]
        self.stride = [int(patch_shape[0]*stride), int(patch_shape[1]*stride),
                       patch_shape[2]]

        self.spatial_jitter = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomResizedCrop(patch_shape[0], scale=(0.7, 0.9))
        ])

    def __call__(self, x):
        if torch.is_tensor(x):
            x = x.numpy()
        elif 'PIL' in str(type(x)):
            x = np.array(x)
        
        if x.shape[0] == 3:
            x = x.transpose(1, 2, 0)
        
        winds = skimage.util.view_as_windows(x, self.patch_shape, step=self.stride)
        winds = winds.reshape(-1, *winds.shape[-3:])

        P = [self.transform(self.spatial_jitter(w)) for w in winds]
        return torch.cat(P, dim=0)

class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])
        
        if isinstance(vid, torch.Tensor):
            vid = vid.numpy()

        if self.pil_convert:
            x = np.stack([np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])

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

    if 'grid' in aug_list:
        transf.append(CropPatches(norm_aug))
    else:
        transf += norm_aug

    return MapTransform(transforms.Compose(transf))
