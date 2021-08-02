import random
import torch

from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

class CUB200(Dataset):
    def __init__(self, path, transform=None, img_size=(600, 600), mode=None):
        self.path = path
        self.mode = mode
        self.img_size = img_size

        # images paths
        with open(f'{path}/images.txt', 'r') as f:
            img_paths = [x.split() for x in f.read().splitlines()]
        # labels
        with open(f'{path}/image_class_labels.txt', 'r') as f:
            img_labels = [int(x.split()[-1]) for x in f.read().splitlines()]
        # bounding boxes (xmin, ymin, w, h)
        with open(f'{path}/bounding_boxes.txt', 'r') as f:
            img_boxes = [list(map(float, x.split()[1:])) for x in f.read().splitlines()]
        
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
            return {'img': img, 'cls': label, 'rot_label': rot_label,
                    'path': self.img_files[index]}
        elif self.mode == 'Localize':
            box = torch.FloatTensor(self.bounding_boxes[index])
            return {'img': img, 'cls': label, 'box': box,\
                    'path': self.img_files[index]}
        
        return {'img': img, 'cls': label, 'path': self.img_files[index]}

    def get_img(self, index):
        image_file = self.img_files[index]
        img = Image.open(image_file).resize(self.img_size).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        imgs = [s['img'] for s in batch]
        target = [s['cls'] for s in batch]
        path = [s['path'] for s in batch]

        imgs = torch.stack(imgs, 0)
        target = torch.cat(target, 0)

        if 'rot_label' in batch[-1]:
            rot_target = [s['rot_label'] for s in batch]
            rot_target = torch.cat(rot_target, 0)
            return path, imgs, target.type(torch.LongTensor),\
                    rot_target.type(torch.LongTensor)
        elif 'box' in batch[-1]:
            boxes = [s['box'] for s in batch]
            boxes = torch.stack(boxes, 0)
            return path, imgs, target.type(torch.LongTensor),\
                    boxes.type(torch.LongTensor)

        return path, imgs, target.type(torch.LongTensor)
