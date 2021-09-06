import argparse
import os
import pickle
import math
import numpy as np
import torch

from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm import tqdm
from IPython import display

from models.crw import CRW
from data.datasets import DAVIS

def get_args():
    parser = argparse.ArgumentParser('Label propagation')
    parser.add_argument('--data-path', type=str, help='path dataset')
    parser.add_argument('--weight-path', type=str, help='path weights')

    parser.add_argument('--depth', type=int, default=18, help='depth resnet model')
    parser.add_argument('--head-depth', type=int, default=0, help='depth head')
    parser.add_argument('--temperature', type=float, default=0.05, help='(temperature) shaping')
    parser.add_argument('--featdrop', type=float, default=0.1, help='dropout rate on maps')
    parser.add_argument('--edgedrop', type=float, default=0.0, help='dropout rate on A')

    parser.add_argument('--img-size', nargs='+', type=int, default=[256, 256], help='image size')
    parser.add_argument('--video-len', type=int, default=20, help='number of context frames')
    parser.add_argument('--patch-size', nargs='+', type=int, default=[64, 64], help='patch size')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--fs', type=int, default=5, help='batch frame size')
    
    parser.add_argument('--n_work', type=int, default=2, help='number of gpu')
    parser.add_argument('--device', type=str, default='cuda', help='use cpu or cuda')

    args = parser.parse_args()
    return args

def test(model, dataloader, opt):
    #
    for imgs, imgs_orig, masks, mask_set, meta in dataloader:
        imgs = imgs.to(opt.device)
        B, T, C, W, H = imgs.shape

        with torch.no_grad():
            embeds = []
            for split in range(0, T, opt.fs):
                batch_imgs = imgs[:, split:split+opt.fs].transpose(1, 2). \
                    to(opt.device) # (B, C, opt.fs, H, W)
                batch_imgs = batch_imgs.permute(0, 2, 1, 3, 4).contiguous(). \
                    view(-1, C, H, W) # (B*opt.fs, C, H, W)
                embed = model(batch_imgs) # (B*opt.fs, c, h, w)
                _, c, h, w = embed.shape
                embed = embed.view(B, T, c, h, w).permute(0, 2, 1, 3, 4)
                embeds.append(embed.cpu())
            embeds = torch.cat(embeds, dim=2).squeeze(1)
            embeds = torch.nn.functional.normalize(embeds, dim=1)
        
        # 
        torch.cuda.empty_cache()
        

if __name__ == '__main__':
    opt = get_args()
    
    # load dataset
    dataset = DAVIS(opt)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=opt.bs, shuffle=False,
                    num_workers=opt.n_work, pin_memory=True)
    
    # create crw model
    model = CRW(opt).to(opt.device)
    
    # load checkpoing weights
    checkpoint = torch.load(opt.weight_path)
    model.load_state_dict(checkpoint['model'])
    
    test(model.encoder, dataloader, opt)
