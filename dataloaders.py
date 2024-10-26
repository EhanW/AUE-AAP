import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import CIFAR10PairIndex, CIFAR100PairIndex, TinyImageNetPairIndex, MiniImageNetPairIndex


class PrepareDataLoaders:
    def __init__(self, dataset: str, root: str, output_size: int, for_gen: bool, supervised: bool,
                 post_aug: bool = False, strength: float = 1.0, delta: torch.FloatTensor = None, ratio=1.0,
                 ):
        self.dataset = dataset
        self.root = root
        self.for_gen = for_gen
        self.supervised = supervised
        self.output_size = output_size
        self.post_aug = post_aug
        self.strength = strength
        self.delta = delta
        self.ratio = ratio


    def get_train_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_train_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return dataloader

    def get_test_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_test_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return dataloader

    def get_val_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_val_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return dataloader

    def _get_train_transform_for_generation(self):
        if self.supervised:
            if not self.post_aug:
                transform = transforms.ToTensor()
            else:
                s = self.strength
                transform = transforms.Compose([transforms.RandomResizedCrop(self.output_size, scale=(1 - 0.9 * s, 1.0)),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomApply(
                                                   [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)],
                                                   p=0.8 * s),
                                                transforms.RandomGrayscale(p=0.2 * s),
                                                transforms.ToTensor()])
        else:
            transform = transforms.ToTensor()
        return transform

    def _get_train_transform_for_evaluation(self):
        if self.supervised:
            if self.dataset in ['cifar10', 'cifar100']:
                transform = transforms.Compose([transforms.RandomCrop(32, 4),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor()])
            elif self.dataset in ['tinyimagenet', 'miniimagenet']:
                transform = transforms.Compose([transforms.RandomCrop(self.output_size, int(self.output_size/8)),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor()])
            elif self.dataset == 'imagenet100':
                transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor()])
            else:
                raise AssertionError('dataset is not defined')
        else:
            transform = transforms.Compose([transforms.RandomResizedCrop(self.output_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                                            p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor()])
        return transform

    def _get_test_transform(self):
        if self.dataset in ['cifar10', 'cifar100', 'tinyimagenet', 'miniimagenet']:
            transform = transforms.ToTensor()
        else:
            raise AssertionError('dataset is not defined')
        return transform

    def _make_dataset(self, train: bool, transform, pair: bool, delta):
        if self.dataset == 'cifar10':
            data_set = CIFAR10PairIndex(root=self.root, train=train, transform=transform, pair=pair, delta=delta,
                                        ratio=self.ratio, download=True)
        elif self.dataset == 'cifar100':
            data_set = CIFAR100PairIndex(root=self.root, train=train, transform=transform, pair=pair,
                                         delta=delta, ratio=self.ratio, download=True)
        elif self.dataset == 'tinyimagenet':
            data_set = TinyImageNetPairIndex(root=self.root, train=train, transform=transform,
                                             pair=pair, delta=delta)
        elif self.dataset == 'miniimagenet':
            data_set = MiniImageNetPairIndex(root=self.root, train=train, transform=transform,
                                             pair=pair, delta=delta)
        else:
            raise AssertionError('dataset is not defined')
        return data_set

    def _get_train_set(self):
        if self.for_gen:
            transform = self._get_train_transform_for_generation()
        else:
            transform = self._get_train_transform_for_evaluation()

        if (not self.for_gen) and (not self.supervised):
            pair = True
        else:
            pair = False

        dataset = self._make_dataset(train=True, transform=transform, pair=pair, delta=self.delta)
        return dataset

    def _get_test_set(self):
        transform = self._get_test_transform()
        pair = False
        dataset = self._make_dataset(train=False, transform=transform, pair=pair, delta=None)
        return dataset

    def _get_val_set(self):
        transform = self._get_test_transform()
        pair = False
        dataset = self._make_dataset(train=True, transform=transform, pair=pair, delta=self.delta)
        return dataset




class APDataLoaders:
    def __init__(self, dataset: str, root: str, output_size: int, ref_mode='standard', rrc=0.0, cj=0.0, rg=0.0):
        self.dataset = dataset
        self.root = root
        self.ref_mode = ref_mode
        self.output_size = output_size
        self.rrc = rrc
        self.cj = cj
        self.rg = rg

    def get_train_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_train_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return dataloader

    def get_test_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_test_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        return dataloader

    def get_val_loader(self, batch_size: int, num_workers: int):
        dataset = self._get_val_set()
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        return dataloader
        
    def _get_train_transform_for_generation(self):
        if self.ref_mode == 'standard':
            transform = transforms.Compose([transforms.RandomCrop(self.output_size, int(self.output_size/8)),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor()])
        else: 
            transform = transforms.Compose([transforms.RandomResizedCrop(self.output_size, scale=(1 - 0.9 * self.rrc, 1.0)),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomApply(
                                                   [transforms.ColorJitter(0.4 * self.cj, 0.4 * self.cj, 0.4 * self.cj, 0.1 * self.cj)],
                                                   p=0.8 * self.cj),
                                                transforms.RandomGrayscale(p=0.2 * self.rg),
                                                transforms.ToTensor()])
        return transform

    def _get_test_transform(self):
        if self.dataset in ['cifar10', 'cifar100', 'tinyimagenet', 'miniimagenet']:
            transform = transforms.ToTensor()
        else:
            raise AssertionError('dataset is not defined')
        return transform

    def _make_dataset(self, train: bool, transform):
        if self.dataset == 'cifar10':
            data_set = CIFAR10PairIndex(root=self.root, train=train, transform=transform, download=True, pair=False)
        elif self.dataset == 'cifar100':
            data_set = CIFAR100PairIndex(root=self.root, train=train, transform=transform, download=True, pair=False)
        elif self.dataset == 'tinyimagenet':
            data_set = TinyImageNetPairIndex(root=self.root, train=train, transform=transform, pair=False)
        elif self.dataset == 'miniimagenet':
            data_set = MiniImageNetPairIndex(root=self.root, train=train, transform=transform, pair=False)
        else:
            raise AssertionError('dataset is not defined')
        return data_set

    def _get_train_set(self):
        transform = self._get_train_transform_for_generation()

        dataset = self._make_dataset(train=True, transform=transform)
        return dataset

    def _get_test_set(self):
        transform = self._get_test_transform()
        dataset = self._make_dataset(train=False, transform=transform)
        return dataset

    def _get_val_set(self):
        transform = self._get_test_transform()
        dataset = self._make_dataset(train=True, transform=transform)
        return dataset
