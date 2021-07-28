import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from IPython import display
from tqdm import tqdm

from models.resnet import resnet50
from models.rotnet import RotNet
from data.dataloaders import CUB200

from tools.utils import get_scheduler, adjust_learning_rate

def get_args():
    parser = argparse.ArgumentParser('Training RotNet for CUB200 - Vlad15lav')
    parser.add_argument('-p', '--path', type=str, help='path dataset')
    parser.add_argument('--sw', '--save_weight', type=str, default='drive/MyDrive' help='save weight')
    parser.add_argument('--lw', '--load_weight', type=str, help='load weight')
    parser.add_argument('--seed', type=int, default=555, help='seed number')

    # training args
    parser.add_argument('--epoches', type=int, default=800, help='number of epoches')
    parser.add_argument('--adam', help='Adam optimizer', action="store_true")
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_work', type=int, default=2, help='number of gpu')
    parser.add_argument('--lr', type=float, default=1e-3, help='init learning rate')
    parser.add_argument('--finlr', type=float, default=1e-4, help='final learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--wlr', type=float, default=1e-5, help='start warmup learning rate')
    parser.add_argument('--w_epoches', type=int, default=10, help='warmup epochs')


    
    parser.add_argument('--load_train', help='continue training', action="store_true")
    parser.add_argument('--debug', help='debug training', action="store_true")

    args = parser.parse_args()
    return args


def main():
	opt = get_args()

	np.random.seed(opt.seed)

	# create and load model
	model = RotNet(resnet50(), num_class=4).cuda()
	if opt.lw:
		model.load_state_dict(torch.load(opt.lw))

    # augmentation
    train_augmentations = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    # create dataloader
    trainset = CUB200(opt.path, train_augmentations)

	dataset_size = len(trainset)
	train_indicies = np.arange(dataset_size)
	np.random.seed(555)
	np.random.shuffle(train_indicies)

	train_sampler = SubsetRandomSampler(train_indicies[:int(dataset_size * 0.8)])
	validation_sampler = SubsetRandomSampler(train_indicies[int(dataset_size * 0.8):int(dataset_size * 0.9)])

	TrainLoader = DataLoader(trainset, batch_size=opt.batch_size, sampler=train_sampler,
		num_workers=2, collate_fn=trainset.collate_fn)
	ValidLoader = DataLoader(trainset, batch_size=128, sampler=validation_sampler,
		num_workers=2, collate_fn=trainset.collate_fn)

	# optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,
            weight_decay=opt.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr,
            momentum=0.9)
    # criterion
	criterion = nn.CrossEntropyLoss()
    # scheduler
    lr_schedule = get_sheduler(opt.lr, opt.finlr, len(TrainLoader), opt.epoches, opt.w_epoches, opt.wlr)

    # training
    loss_train = []
	acc_valid = []

	for epoch in tqdm(range(opt.epochs)):
	    display.clear_output(wait=True)
	    
	    loss_epoch = []
	    model.train()
	    for i, (_, imgs, labels) in enumerate(TrainLoader):
	        adjust_learning_rate(optimizer, lr_schedule, iteration=opt.epochs * len(TrainLoader) + i)
	        optimizer.zero_grad()

	        imgs = Variable(imgs.cuda(), requires_grad=True)
	        labels = Variable(labels.cuda(), requires_grad=False)

	        out = model.forward(imgs)
	        loss = criterion(out, labels)

	        loss.backward()
	        optimizer.step()
	        
	        loss_epoch.append(float(loss))
	    loss_train.append(np.mean(loss_epoch))

	    # validation
	    model.eval()
	    accuracy = 0
	    for _, imgs, labels in ValidLoader:
	        with torch.no_grad():
	            imgs = Variable(imgs.cuda(), requires_grad=False)
	            out = model.forward(imgs)
	            
	            accuracy += sum(out.cpu().clone().detach().argmax(axis=1) == labels).item()

	    acc_valid.append(accuracy / (len(ValidLoader) * opt.batch_size))

	    # save best and last weight
	    if len(acc_valid) > 1 and acc_valid[-1] > np.max(acc_valid[:-1]):
	       torch.save(model.state_dict(), f'{opt.sw}/RotNet_Best.pth')
	    torch.save(model.state_dict(), f'{opt.sw}/RotNet_Last.pth')

	    # ploting training
	    _, axes = plt.subplots(1, 2, figsize=(12, 6))
	    axes[0].set_title("Train Set")
	    axes[0].plot(loss_train, label="Loss")
	    axes[0].legend()
	    
	    axes[1].set_title("Valid Set")
	    axes[1].plot(acc_valid, label="Accuracy")
	    axes[1].legend()
	    plt.show()


if __name__ == '__main__':
    main()
