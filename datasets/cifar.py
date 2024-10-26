import torch
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from PIL import Image


class CIFAR10PairIndex(CIFAR10):
    def __init__(self, pair=True, delta: torch.FloatTensor = None, ratio=1.0, **kwargs):
        super(CIFAR10PairIndex, self).__init__(**kwargs)
        self.delta = delta
        self.pair = pair

        assert ratio <= 1.0 and ratio > 0
        if self.delta is not None:
            if len(delta) == 10:
                self.delta = self.delta[torch.tensor(self.targets)]
            if delta.shape != self.data.shape:
                self.delta = self.delta.permute(0, 2, 3, 1)
                assert self.delta.shape == self.data.shape

            set_size = int(len(self.data) * ratio)
            if set_size < len(self.data):
                self.delta[set_size:] = 0.0
            self.delta = self.delta.mul(255).cpu().numpy()
            self.data = np.clip(self.data.astype(np.float32) + self.delta, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.pair:
            img = torch.stack([self.transform(img), self.transform(img)], dim=0)
        else:
            img = self.transform(img)
        return img, target, idx


class CIFAR100PairIndex(CIFAR100):
    def __init__(self, pair=True, delta: torch.FloatTensor = None, ratio=1.0, **kwargs):
        super(CIFAR100PairIndex, self).__init__(**kwargs)
        self.delta = delta
        self.pair = pair

        if self.delta is not None:
            if len(delta) == 100:
                self.delta = self.delta[torch.tensor(self.targets)]
            if delta.shape != self.data.shape:
                self.delta = self.delta.permute(0, 2, 3, 1)
                assert self.delta.shape == self.data.shape
            set_size = int(len(self.data) * ratio)
            if set_size < len(self.data):
                self.delta[set_size:] = 0.0
            self.delta = self.delta.mul(255).cpu().numpy()
            self.data = np.clip(self.data.astype(np.float32) + self.delta, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        if self.pair:
            img = torch.stack([self.transform(img), self.transform(img)], dim=0)
        else:
            img = self.transform(img)
        return img, target, idx

