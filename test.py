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
    parser.add_argument('--bs', type=int, default=8, help='batch size')
    
    parser.add_argument('--n_work', type=int, default=2, help='number of gpu')
    parser.add_argument('--device', type=str, default='cuda', help='use cpu or cuda')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_args()
    
    # load dataset
    dataset = DAVIS(opt)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=opt.bs, shuffle=False
                    num_workers=opt.n_work, pin_memory=True)
    
    # create crw model
    model = CRW(opt).to(opt.device)
    
    # load checkpoing weights
    checkpoint = torch.load(opt.weight_path)
    model.load_state_dict(checkpoint['model'])
    
    # TODO: Label prob test
