import numpy as np
import torch
import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset


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
