import random
import torch

from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

class CUB200(Dataset):
	"""
	Dataset class for RotNet
	"""
    def __init__(self, path, transform=None):
        self.path = path
        with open(f'{path}/images.txt', 'r') as f:
            img_paths = [x.split() for x in f.read().splitlines()]

        img_files = []
        for i in range(len(img_paths)):
            img_files.append(self.path + '/images/' + img_paths[i][1])
        
        self.img_files = img_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = self.get_img(index)
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = random.randint(0, 3)

        img = transforms.functional.rotate(img, 90 * label)
        label = torch.FloatTensor([label])
        
        return img, label, self.img_files[index]

    def get_img(self, index):
        image_file = self.img_files[index]
        img = Image.open(image_file).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        img, target, path = zip(*batch)
        img = torch.stack(img, 0)
        target = torch.cat(target, 0)
        return path, img, target.type(torch.LongTensor)
