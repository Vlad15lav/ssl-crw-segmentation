import skimage
import torch
import torchvision
import numpy as np

from torchvision import transforms
from PIL import Image

def get_train_augmentation(opt):
    resize_aug = transforms.Resize(opt.img_size)
    norm_aug = [transforms.ToTensor(), 
        transforms.Normalize(opt.mean_norm, opt.std_norm)]
    
    
