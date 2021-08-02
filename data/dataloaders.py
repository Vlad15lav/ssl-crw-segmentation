import random
import torch

from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

class CUB200(Dataset):
    def __init__(self, path, transform=None, mode=None):
        self.path = path
        self.mode = mode
        # images paths
        with open(f'{path}/images.txt', 'r') as f:
            img_paths = [x.split() for x in f.read().splitlines()]
        # labels
        with open(f'{path}/image_class_labels.txt', 'r') as f:
            img_labels = [int(x.split()[-1]) for x in f.read().splitlines()]
        # bounding boxes (xmin, ymin, w, h)
        with open(f'{path}/bounding_boxes.txt', 'r') as f:
            img_boxes = [int(x.split()[1:]) for x in f.read().splitlines()]
        
        img_files, all_targets, bounding_boxes = [], [], []
        for i in range(len(img_paths)):
            img_files.append(self.path + '/images/' + img_paths[i][1])
            all_targets.append(img_labels[i] - 1)
            bounding_boxes.append(img_boxes[i])
        
        self.img_files = img_files
        self.all_targets = all_targets
        self.bounding_boxes = bounding_boxes
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = self.get_img(index)
        
        label = self.all_targets[index]
        label = torch.FloatTensor([label])
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.mode == 'RotNet':
            rot_label = random.randint(0, 3)
            img = transforms.functional.rotate(img, 90 * rot_label)
            rot_label = torch.FloatTensor([rot_label])
            return img, label, rot_label, self.img_files[index]
        elif self.mode == 'Localize':
            return img, label, self.bounding_boxes[index] self.img_files[index]
        
        return img, label, self.img_files[index]

    def get_img(self, index):
        image_file = self.img_files[index]
        img = Image.open(image_file).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        img, target, path = zip(*batch)
        if self.mode == 'RotNet':
            img, target, rot_target, path = zip(*batch)
            rot_target = torch.cat(rot_target, 0)
        elif self.mode == 'Localize':
            img, target, boxes, path = zip(*batch)
            boxes = torch.stack(boxes, 0)
        
        img = torch.stack(img, 0)
        target = torch.cat(target, 0)
        
        if self.mode == 'RotNet':
            path, img, target.type(torch.LongTensor), rot_target.type(torch.LongTensor)
        elif self.mode == 'Localize':
            path, img, target.type(torch.LongTensor), boxes.type(torch.LongTensor)
        return path, img, target.type(torch.LongTensor)
