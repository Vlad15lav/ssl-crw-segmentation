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

from data.datasets import Kinetics400
from data.augments import get_train_augmentation
from models.crw import CRW
from utils.util import get_cache_path, collate_fn, get_sheduler, adjust_learning_rate

def get_args():
    parser = argparse.ArgumentParser('Training CRW')
    parser.add_argument('--data-path', type=str, help='path dataset')
    parser.add_argument('--weight-path', type=str, default='state/crw', help='path for logs, weights training')
    parser.add_argument('--name-checkpoint', type=str, default='checkpoint', help='name checkpoint file pth')
    parser.add_argument('--cache-path', type=str, help='cache file dataset')

    parser.add_argument('--depth', type=int, default=18, help='depth resnet model')
    parser.add_argument('--head-depth', type=int, default=0, help='depth head')
    parser.add_argument('--pretrained', help='load imagenet weights', action="store_true")
    parser.add_argument('--cont-train', help='use last weights', action="store_true")
    parser.add_argument('--temperature', type=float, default=0.05, help='(temperature) shaping')
    parser.add_argument('--featdrop', type=float, default=0.1, help='dropout rate on maps')
    parser.add_argument('--edgedrop', type=float, default=0.0, help=' dropout rate on A')

    parser.add_argument('--img-size', nargs='+', type=int, default=[256, 256], help='image size')
    parser.add_argument('--clip-len', default=4, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--frame-skip', default=8, type=int, help='kinetics: fps | others: skip between frames')
    parser.add_argument('--augs', type=str, default='crop', help='select augmentation (crop, jitter, flip, grid)')
    parser.add_argument('--patch-size', nargs='+', type=int, default=[64, 64], help='patch size')
    parser.add_argument('--mean-norm', nargs='+', type=int, default=[0.4914, 0.4822, 0.4465], help='mean pixel')
    parser.add_argument('--std-norm', nargs='+', type=int, default=[0.2023, 0.1994, 0.2010], help='std pixel')
    parser.add_argument('--bs', type=int, default=8, help='batch size')
    parser.add_argument('--epoches', type=int, default=40, help='number of epoches')
    
    parser.add_argument('--lr', type=float, default=0.0002, help='init learning rate')
    parser.add_argument('--final-lr', type=float, default=0.00005, help='init learning rate')
    parser.add_argument('--wup-lr', type=float, default=0.000001, help='start learning rate warm-up')
    parser.add_argument('--warm-up', type=int, default=1, help='number of epoche with warm-up')
    parser.add_argument('--wd', '--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--adam', help='adam optimizer', action="store_true")
    parser.add_argument('--n_work', type=int, default=2, help='number of gpu')
    parser.add_argument('--seed', type=int, default=841, help='number of seed')
    parser.add_argument('--device', type=str, default='cuda', help='use cpu or cuda')

    args = parser.parse_args()
    return args


def train(model, train_loader, valid_loader, optimizer, lr_schedule, opt):
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    
    # load checkpoing weights
    if os.path.exists(opt.weight_path) and opt.cont_train:
        checkpoint = torch.load(f'{opt.weight_path}/{opt.name_checkpoint}.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('weights loaded!')
    
    if os.path.exists(f'{opt.weight_path}/log_training.pickle'):
        f_log = open(f'{opt.weight_path}/log_training.pickle', 'rb')
        obj = pickle.load(f_log)
        train_loss, train_acc, valid_loss, valid_acc, valid_loss, valid_acc = obj
        f_log.close()
        print('training data loaded!')

    print("Training start...")
    train_size = len(train_loader)
    for epoch in range(len(train_loss), opt.epoches):
        display.clear_output(wait=True)
        
        # training
        model.train()
        loss_batch, acc_batch = [], []
        for i, clip in enumerate(train_loader):
            
            adjust_learning_rate(optimizer, lr_schedule, epoch * train_size + i)
            optimizer.zero_grad()

            clip = Variable(clip.to(opt.device))
            q, loss, acc = model(clip)

            loss.backward()
            optimizer.step()

            loss_batch.append(loss.item())
            acc_batch.append(acc.cpu())
            
            if (i + 1) % int(train_size * 0.025) == 0:
                print(f"\tProgress training of epoche - {(i + 1) * 100 // train_size}%", end=' | ')
                print(f"train loss - {np.mean(loss_batch)}, train acc - {np.mean(acc_batch)}")
                # save last and best weights
                checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}
                torch.save(checkpoint, os.path.join(opt.weight_path, 'iter_checkpoint.pth'))
        
        train_loss.append(np.mean(loss_batch))
        train_acc.append(np.mean(acc_batch))
        
        # validation loss
        model.eval()
        loss_batch, acc_batch = [], []
        with torch.no_grad():
            for i, clip in enumerate(valid_loader):
                clip = Variable(clip.to(opt.device))
                q, loss, acc = model(clip)
                
                loss_batch.append(loss.item())
                acc_batch.append(acc.cpu())
        
        valid_loss.append(np.mean(loss_batch))
        valid_acc.append(np.mean(acc_batch))
        
        # print status training
        print(f'(epoche {epoch + 1}): train loss: {train_loss[-1]}, train accuracy: {train_acc[-1]}', end=', ')
        print(f'valid loss: {valid_loss[-1]}, valid accuracy: {valid_acc[-1]}')
        
        if len(valid_loss) > 1 and valid_loss[-1] < np.min(valid_loss[:-1]):
            # save last and best weights
            checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}
            torch.save(
                    checkpoint,
                    os.path.join(opt.weight_path, 'best_checkpoint.pth'))

        # save last and best weights
        checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}
        torch.save(
                checkpoint,
                os.path.join(opt.weight_path, 'checkpoint.pth'))
 
        # log history
        lists = (train_loss, train_acc, valid_loss, valid_acc)
        f_log = open(f'{opt.weight_path}/log_training.pickle', 'wb')
        pickle.dump(lists, f_log)
        f_log.close()

if __name__ == '__main__':
    opt = get_args()
    
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if not os.path.exists(opt.weight_path):
        os.makedirs(opt.weight_path)

    # get transfroms for dataloader
    transform_train = get_train_augmentation(opt)
    
    # load cache of dataset
    cached = None
    if opt.cache_path:
        if not os.path.exists(opt.cache_path):
            os.makedirs(opt.cache_path)
        
        cache_path = get_cache_path(opt.cache_path)
        if os.path.exists(cache_path):
            trainset, _ = torch.load(cache_path)
            cached = dict(video_paths=trainset.video_clips.video_paths,
                    video_fps=trainset.video_clips.video_fps,
                    video_pts=trainset.video_clips.video_pts)

    # load dataset
    trainset = Kinetics400(root=opt.data_path + '/train',
                           frames_per_clip=opt.clip_len,
                           step_between_clips=1,
                           transform=transform_train,
                           extensions=('mp4'),
                           frame_rate=opt.frame_skip,
                           _precomputed_metadata=cached)
    validset = Kinetics400(root=opt.data_path + '/valid',
                           frames_per_clip=opt.clip_len,
                           step_between_clips=1,
                           transform=transform_train,
                           extensions=('mp4'),
                           frame_rate=opt.frame_skip,
                           _precomputed_metadata=None)
    
    # save cache dataset
    if cached is None and cache_path:
        trainset.transform = None
        torch.save((trainset, opt.data_path), cache_path)
        trainset.transform = transform_train
    
    # create dataloader
    train_sampler = RandomSampler(trainset)
    if os.path.exists(opt.weight_path) and opt.cont_train:
        checkpoint = torch.load(f'{opt.weight_path}/{opt.name_checkpoint}.pth')
        train_sampler = checkpoint['state_sampler']
    
    train_loader = DataLoader(trainset, batch_size=opt.bs, sampler=train_sampler,
                    num_workers=opt.n_work, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(validset, batch_size=opt.bs,
                    num_workers=opt.n_work, pin_memory=True, collate_fn=collate_fn)
    
    # create crw model
    model = CRW(opt).to(opt.device)
    
    # optimizer and sheduler
    lr_schedule = get_sheduler(opt.lr, opt.final_lr, len(train_loader), opt.epoches, opt.warm_up, opt.wup_lr)

    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,
            weight_decay=opt.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr,
            momentum=opt.momentum, weight_decay=opt.wd)
    
    train(model, train_loader, valid_loader, optimizer, lr_schedule, opt)
