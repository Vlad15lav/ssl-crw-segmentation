import os
import numpy as np
import torch
import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

from utils.tools import *

class Kinetics400(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`
    Arguments:
        path(str) - path directory of the Kinetics-400 dataset
        frames_per_clip(int) - number of frames in a clip
        step_between_clips(int) - number of frames between each clip
        transform(callable, optional) - torchvision transforms for TxHxWxC video
    Returns:
        video(Tensor[T, H, W, C]) - T - frame, H - height, W - width, C - channel
        audio(Tensor[K, L]) - K - number of channels, L - number of points
        label(int) - class of the video clip
    """
    def __init__(self, root, frames_per_clip, step_between_clips=1, 
                  frame_rate=None, extensions=('mp4',),
                  transform=None, cached=None, _precomputed_metadata=None):
        super(Kinetics400, self).__init__(root)
        extensions = extensions

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
        )
        self.transform = transform

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)
                success = True
            except:
                idx = np.random.randint(self.__len__())

        label = self.samples[video_idx][1]
        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label

class DAVIS(torch.utils.data.Dataset):
    """
    https://davischallenge.org/
    """
    def __init__(self, opt):
        self.path = opt.data_path
        self.img_size = opt.img_size
        self.video_len = opt.video_len
        
        self.images_path = os.path.join(self.path, 'JPEGImages', '480p')
        self.masks_path = os.path.join(self.path, 'Annotations', '480p')
        self.classes_list = os.listdir(self.images_path)

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.classes_list)

    def num_classes(self):
        return len(self.classes_list)

    def label_to_name(self, num_class):
        return self.classes_list[num_class]

    def get_paths(self, folder_imgs, folder_masks):
        img_files, mask_files = os.listdir(folder_imgs), os.listdir(folder_masks)

        frame_num = len(img_files) + self.video_len
        img_files.sort(key=lambda x:int(x.split('.')[0]))
        mask_files.sort(key=lambda x:int(x.split('.')[0]))

        imgs_paths, masks_paths = [], []

        for i in range(frame_num):
            i = max(0, i - self.video_len)
            img_path = os.path.join(folder_imgs, img_files[i])
            mask_path = os.path.join(folder_masks,  mask_files[i])

            imgs_paths.append(img_path)
            masks_paths.append(mask_path)

        return imgs_paths, masks_paths

    def load_image(self, path):
        img = cv2.imread(path)
        img = img.astype(np.float32) / 255.
        img = img[:, :, ::-1]
        img = img.copy()
        img = np.transpose(img, (2, 0, 1)) # CxHxW
        img = torch.from_numpy(img).float()
        return img

    def __getitem__(self, index):
        class_imgs = os.path.join(self.images_path, self.classes_list[index])
        class_masks = os.path.join(self.masks_path, self.classes_list[index])

        frame_num = len(os.listdir(class_imgs)) + self.video_len
        imgs_paths, masks_paths = self.get_paths(class_imgs, class_masks)

        imgs = []
        imgs_orig = []
        masks = []

        for i in range(frame_num):
            img_path, mask_path = imgs_paths[i], masks_paths[i]
            img = self.load_image(img_path)
            mask = cv2.imread(mask_path)

            # resize image and mask
            img = resize(img, *self.img_size)
            mask = cv2.resize(mask, self.img_size, cv2.INTER_NEAREST)

            img_orig = img.clone()

            # normalize pixels
            img = color_normalize(img, self.mean, self.std)

            imgs.append(img)
            imgs_orig.append(img_orig)
            masks.append(mask)
        
        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        masks = np.stack(masks)

        # mask set channel
        mask_set = masks[0].copy().reshape(-1, masks.shape[-1]).astype(np.uint8)
        mask_set = np.unique(mask_set, axis=0)

        if np.all((mask_set[1:] - mask_set[:-1]) == 1):
            mask_set = mask_set[:, :1]
        
        meta = {'path_images': class_imgs, 'path_masks': class_masks, 'class':self.classes_list[index]}
        return imgs, imgs_orig, masks, mask_set, meta
