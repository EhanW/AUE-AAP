import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
import os
from kornia import augmentation as KA
from torch.nn import functional as F
import random


class Augment(object):
    def __init__(self, strength=1, size=32):
        self.rrc = KA.RandomResizedCrop(size=(size, size), scale=(max(1-0.9*strength, 0.05), 1))
        self.rhf = KA.RandomHorizontalFlip(p=0.5)
        self.cj = KA.ColorJitter(brightness=0.4*strength, contrast=0.4*strength, saturation=0.4*strength, hue=0.1*strength, p=min(0.99, 0.8*strength))
        self.rg = KA.RandomGrayscale(p=0.2*strength)
        self.rc = KA.RandomCrop((size, size), int(size/8))

    def aug_cl(self, data):
        img = self.rrc(data)
        img = self.rhf(img)
        img = self.cj(img)
        img = self.rg(img)
        return img
    
    def aug_standard(self, data):
        img = self.rc(data)
        img = self.rhf(img)
        return img

    def aug_id(self, data):
        return data


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=.9,
                 weight_decay=.0005, eta=0.001):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta)
        super(LARS, self).__init__(params, defaults)

    def step(self, gradMulti=1, epoch=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            # max_epoch = group['max_epoch']

            for cnt, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data * gradMulti

                # if cnt == 0:
                #     print("d_p is {}".format(d_p * gradMulti))

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # Global LR computed on polynomial decay schedule
                # decay = (1 - float(epoch) / max_epoch) ** 2
                global_lr = lr

                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / \
                    (grad_norm + weight_decay * weight_norm)

                # if len(local_lr[(weight_norm < 1e-15) | (grad_norm < 1e-15)]) > 0:
                #     print("len zeros is {}".format(len(local_lr[(weight_norm < 1e-15) | (grad_norm < 1e-15)])))
                local_lr[(weight_norm < 1e-15) | (grad_norm < 1e-15)] = 1.0

                # Update the momentum term
                actual_lr = local_lr * global_lr

                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = \
                            torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(actual_lr, d_p + weight_decay * p.data)
                p.data.add_(-buf)

        return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class logger(object):
    def __init__(self, path, name='log.txt'):
        self.path = path
        self.name = name

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, self.name), 'a') as f:
            f.write(msg + "\n")

def pair_cosine_similarity(x, y=None, eps=1e-8):
    if (y == None):
        n = x.norm(p=2, dim=1, keepdim=True)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps)
    else:
        n1 = x.norm(p=2, dim=1, keepdim=True)
        n2 = y.norm(p=2, dim=1, keepdim=True)
        return (x @ y.t()) / (n1 * n2.t()).clamp(min=eps)

def nt_xent(x, y=None, t=0.5):
    if (y != None):
        x = pair_cosine_similarity(x, y)
    else:
        x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -torch.log(x).mean()

def byol_mse(online_output, target_output):
    online_output = F.normalize(online_output, dim=1, p=2)
    target_output = F.normalize(target_output, dim=1, p=2)

    online1, online2 = online_output[::2], online_output[1::2]
    target1, target2 = target_output[::2], target_output[1::2]
    loss = 2 - 2 * (online1 * target2).sum(dim=1) + 2 - 2 * (online2 * target1).sum(dim=1)
    return loss.mean()

def save_checkpoint(state, filename):
    torch.save(state, filename)

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * (step + 1) / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step -
                                                             warmup_steps) / (total_steps - warmup_steps) * np.pi))
        
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def linear_probing(val_loader, model, optimizer, scheduler, log):
    for epoch in range(100):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        losses = AverageMeter()
        for i, (data, target, _) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            loss = F.cross_entropy(model.eval()(data), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), data.shape[0])
        log.info(f'LINEAR PROBING\t'
                 f'Epoch[{epoch}/100]\t'
                 f'avg loss = {losses.avg:.2f}\t'
                 f'current lr = {current_lr:.2f}')
        scheduler.step()


def evaluation(loader, model):
    top1 = AverageMeter()
    for i, (data, target, _) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            outputs = model.eval()(data)
        prec1 = accuracy(outputs.data, target)[0]
        top1.update(prec1.item(), len(data))
    return top1.avg


def setup_seed(seed):
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Numpy
    np.random.seed(seed)
    # Python
    random.seed(seed)

