import torch
import os
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import glob
from shutil import move
from os import rmdir


def prepare_val_folder(target_folder='./tiny-imagenet-200/val/'):
    val_dict = {}
    with open(os.path.join(target_folder, 'val_annotations.txt'), 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob(os.path.join(target_folder, 'images/*'))
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(os.path.join(target_folder, folder)):
            os.mkdir(os.path.join(target_folder, folder))
            os.mkdir(os.path.join(target_folder, folder, 'images'))

    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = os.path.join(target_folder, folder, 'images',  file)
        move(path, dest)

    rmdir(os.path.join(target_folder, 'images'))


class TinyImageNetPairIndex(ImageFolder):
    def __init__(self, root, train=True, transform=None, pair=True, delta: torch.FloatTensor = None):
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        super().__init__(root, transform=transform)
        self.delta = delta
        self.pair = pair

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.delta is not None:
            if len(self.delta) == 200:
                delta = self.delta[target]
            else:
                delta = self.delta[index]
            delta = delta.mul(255).numpy().transpose(1, 2, 0)
            sample = np.asarray(sample)
            sample = np.clip(sample.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            sample = Image.fromarray(sample, mode='RGB')

        if self.pair:
            sample = torch.stack([self.transform(sample), self.transform(sample)], dim=0)
        else:
            sample = self.transform(sample)
        return sample, target, index
