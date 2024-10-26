import os.path
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image


def prepare_dataset(root: str):
    X = []
    for title in ['train', 'test', 'val']:
        x = open(os.path.join(root, f"mini-imagenet-cache-{title}.pkl"), "rb")
        x = pickle.load(x)
        x = x["image_data"]
        x = x.reshape([-1, 600, 84, 84, 3])
        X.append(x)
    X = np.concatenate(X)
    X_train = X[:, :500, :, :, :].reshape(-1, 84, 84, 3)
    X_test = X[:, 500:, :, :, :].reshape(-1, 84, 84, 3)
    np.save('../data/mini-imagenet/train.npy', X_train)
    np.save('../data/mini-imagenet/test.npy', X_test)


class MiniImageNetPairIndex(Dataset):
    def __init__(self, root: str, train=True, transform=None, pair=True, delta: torch.FloatTensor = None):
        self.labels = []
        if train:
            self.samples = np.load(os.path.join(root, 'train.npy'))
            for i in range(100):
                for j in range(500):
                    self.labels.append(i)
        else:
            self.samples = np.load(os.path.join(root, 'test.npy'))
            for i in range(100):
                for j in range(100):
                    self.labels.append(i)
        self.delta = delta
        self.pair = pair
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img, target = self.samples[index], self.labels[index]
        if self.delta is not None:
            if len(self.delta) == 100:
                delta = self.delta[target]
            else:
                delta = self.delta[index]
            delta = delta.mul(255).numpy().transpose(1, 2, 0)
            img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        if self.pair:
            img = torch.stack([self.transform(img), self.transform(img)], dim=0)
        else:
            img = self.transform(img)
        return img, target, index

