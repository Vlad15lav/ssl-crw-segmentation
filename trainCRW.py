import argparse
import os
import math
import numpy as np
import torch

from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm import tqdm
from IPython import display

from data.datasets import Kinetics400
from models.resnet import get_resnet
from utils.util import collate_fn, get_sheduler, adjust_learning_rate

def get_args():
    parser = argparse.ArgumentParser('Training CRW')
    parser.add_argument('--data-path', type=str, help='path dataset')
    parser.add_argument('--weight-path', type=str, default='state/crw', help='path for logs, weights training')

    parser.add_argument('--depth', type=int, default=18, help='depth resnet model')
    parser.add_argument('--pretrained', help='load imagenet weights', action="store_true")
    parser.add_argument('--cont-train', help='use last weights', action="store_true")

    parser.add_argument('--img_size', type=int, default=256, help='image size')
    parser.add_argument('--bs', type=int, default=8, help='batch size')
    parser.add_argument('--epoches', type=int, default=100, help='number of epoches')
    
    parser.add_argument('--lr', type=float, default=0.005, help='init learning rate')
    parser.add_argument('--final-lr', type=float, default=0.00005, help='init learning rate')
    parser.add_argument('--wup-lr', type=float, default=0.000001, help='start learning rate warm-up')
    parser.add_argument('--warm-up', type=int, default=3, help='number of epoche with warm-up')
    parser.add_argument('--wd', '--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--adam', help='adam optimizer', action="store_true")
    parser.add_argument('--n_work', type=int, default=2, help='number of gpu')
    parser.add_argument('--device', type=str, default='cuda', help='use cpu or cuda')

    args = parser.parse_args()
    return args


def train(model, trainloader, validloader, optimizer, lr_schedule, opt):
    train_loss, val_loss = [], []
    
    if not os.path.exists(opt.weight_path):
        f_log = open(f'{opt.weight_path}/log_training.pickle', 'rb')
        obj = pickle.load(f_log)
        train_loss, val_loss = obj
        f_log.close()
    
    for epoch in tqdm(range(len(train_loss), opt.epoches)):
        #display.clear_output(wait=True)
        
        # training
        model.train()
        loss_batch = []
        for i, (video, orig) in enumerate(trainloader):
            adjust_learning_rate(optimizer, lr_schedule, epoch * len(trainloader) + i)
            optimizer.zero_grad()

            video = Variable(video.to(opt.device))
            output, loss, diagnostics = model(video)
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            loss_batch.append(loss.item())
        train_loss.append(np.mean(loss_batch))
        
        # TODO: Valid

        # print status training
        print(f'epoche {epoch}: train loss {train_loss[-1]}')

        # save last and best weights
        if len(val_loss) > 1 and val_loss[-1] < np.min(val_loss[:-1]):
            torch.save(model.state_dict(), f'{opt.weight_path}/best_weights.pth')
        torch.save(model.state_dict(), f'{opt.weight_path}/last_weights.pth')
 
        # log history
        lists = (train_loss, val_loss)
        f_log = open(f'{opt.weight_path}/log_training.pickle', 'wb')
        pickle.dump(lists, f_log)
        f_log.close()

if __name__ == '__main__':
    opt = get_args()

    if not os.path.exists(opt.weight_path):
        os.makedirs(opt.weight_path)

    # load data loaders
    #transform_train = get_train_transforms()

    trainset = Kinetics400(root=opt.data_path + '/train', frames_per_clip=8, step_between_clips=1,
                            frame_rate=8)#, transform=transform_train)
    # validset = Kinetics400(root=opt.data_path + '/valid', frames_per_clip=8, step_between_clips=1,
    #                         frame_rate=8)#, transform=transform_train)
    train_sampler = RandomSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=opt.bs, sampler=train_sampler,
                    num_workers=opt.n_work, pin_memory=True, collate_fn=collate_fn)
    # validloader = DataLoader(validset, batch_size=opt.batch_size, sampler=train_sampler,
    #                 num_workers=opt.n_work, pin_memory=True, collate_fn=collate_fn)

    # TODO: load model
    # model = get_resnet(opt.depth, opt.pretrained)
    # model.load_state_dict(torch.load(f'{opt.weight_path}/last_weights.pth'))

    # optimizer and sheduler
    lr_schedule = get_sheduler(opt.lr, opt.final_lr, len(trainloader), opt.epoches, opt.warm_up, opt.wup_lr)

    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,
            weight_decay=opt.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr,
            momentum=cfg.momentum, weight_decay=opt.wd)
    
    for video, org in trainloader:
        break
    
    print(video)
    print(org)
    print(len(video))
    print(org)

    #train(model, trainloader, validloader, optimizer, lr_schedule, opt)
