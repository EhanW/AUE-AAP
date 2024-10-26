import torch
import os
import argparse
import time
from utils import AverageMeter, cosine_annealing, logger, accuracy, save_checkpoint, Augment, setup_seed
from torch.nn import functional as F
from torchvision.models import resnet18, resnet50
from models import vgg19, MobileNetV2, DenseNet121
from dataloaders import PrepareDataLoaders
from torch import nn


def get_args():
    parser = argparse.ArgumentParser(description='GEN-AUE')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg19',
                                                                             'mobilenet', 'densenet121'],
                        help='the model arch used in experiment')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                 'miniimagenet'],
                        help='the dataset used in experiment')
    parser.add_argument('--data', type=str, default='data', 
                        help='the directory of dataset')
    parser.add_argument('--num-classes', default=10, type=int, 
                        help='the number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='the batch size used in experiment')
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--mode', type=str, default='constant', choices=['temper', 'anneal', 'constant'],
                        help='the dynamic augmentation scheme. default is constant augmentation scheme')
    parser.add_argument('--cosine-warmup', default=0, type=int,
                        help='the number of warmup steps in cosine tempering dynamic augmentation scheme')
    parser.add_argument('--dynamic-frequency', type=int, default=1,
                        help='the dynamic frequency (interval) in the dynamic scheme')
    parser.add_argument('--strength', type=float, default=1.0,
                        help='the strength of constant augmentation')
    parser.add_argument('--poison-size', type=int, default=32,
                        help='the image size of poisons')
    parser.add_argument('--class-wise', action='store_true',
                        help='if generate class-wise poisons')
    parser.add_argument('--post-aug', action='store_true',
                        help='if generate non-differenitiable poisons')
    parser.add_argument('--num-updates', type=int, default=391,
                        help='the number of model updates in each epoch')
    parser.add_argument('--num-perturbs', type=int, default=391,
                        help='the number of poison updates in each epoch')
    parser.add_argument('--perturb-iters', default=5, type=int,
                        help='the number of PGD steps for updating poisons')
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='the optimizer used in training')
    parser.add_argument('--epochs', default=60, type=int,
                        help='the number of total epochs to run')
    parser.add_argument('--lr', default=0.1, type=float, 
                        help='optimizer learning rate')
    parser.add_argument('--resume', action='store_true', 
                        help='if resume training')
    parser.add_argument('--seed', default=1, type=int, 
                        help='random seed')
    parser.add_argument('--eps', type=int, default=8,
                        help='the L-inf norm constraint for perturbations')
    parser.add_argument('--gpu-id', type=str, default='0', 
                        help='the gpu id')
    return parser.parse_args()


def dynamic_strength(mode, epoch, total_epochs, frequency):
    if mode == 'temper':
        strength = (epoch // frequency) * (1.0 / total_epochs * frequency)
    elif mode == 'anneal':
        strength = 1 - (epoch // frequency) * (1.0 / total_epochs * frequency)
    else:
        strength = args.strength
    return strength


def perturb(train_loader, model, poison, aug):
    model.requires_grad_(False)
    for i, (data, target, index) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        if args.class_wise:
            poison = error_minimizing(model, data, target, index, poison, aug)
        else:
            poison[index] = error_minimizing(model, data, target, index, poison, aug)
        if (i+1) % args.num_perturbs == 0:
            break
    model.requires_grad_(True)
    return poison


def error_minimizing(model, data, target, index, poison, aug):
    eps = args.eps/255
    alpha = eps/10
    iters = args.perturb_iters

    if args.class_wise:
        delta = torch.nn.Parameter(poison)
    else:
        delta = poison[index]
        delta = torch.nn.Parameter(delta)

    for _ in range(iters):
        if args.class_wise:
            inputs = data + delta[target]
        else:
            inputs = data + delta
        img = aug(inputs)
        features = model.eval()(img)
        model.zero_grad()
        loss = F.cross_entropy(features, target)
        loss.backward()

        delta.data = delta.data - alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        if not args.class_wise:
            delta.data = torch.clamp(data + delta.data, min=0, max=1) - data
    return delta.detach()


def train_epoch(train_loader, model, poison, optimizer, scheduler, epoch, log, aug):
    losses = AverageMeter()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    pert_time_meter = AverageMeter()

    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    start = time.time()

    for i, (data, target, index) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        if args.class_wise:
            inputs = aug(data + poison[target])
        else:
            inputs = aug(data + poison[index])
        data_time = time.time() - start
        data_time_meter.update(data_time)

        features = model.train()(inputs)
        loss = F.cross_entropy(features, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), len(data))
        train_time = time.time() - start
        train_time_meter.update(train_time)
        start = time.time()
        if (i+1) == args.num_updates or (i+1) == len(train_loader):
            break

    poison = perturb(train_loader, model, poison, aug)
    pert_time_meter.update(time.time() - start)
    log.info(
        f'Epoch[{epoch}/{args.epochs}]\t'
        f'current lr = {current_lr:.3f}\t'
        f'avg loss = {losses.avg:.4f}\t'
        f'train time = {train_time_meter.sum:.2f}\t'
        f'data time = {data_time_meter.sum:.2f}\t'
        f'pert time = {pert_time_meter.sum:.2f}\t'
        f'epoch time = {train_time_meter.sum+pert_time_meter.sum:.2f}'
    )
    scheduler.step()
    return poison


def evaluation(loader, model):
    top1 = AverageMeter()
    for i, (data, target, _) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            outputs = model.eval()(data)
        prec1 = accuracy(outputs.data, target)[0]
        top1.update(prec1.item(), len(data))
    return top1.avg


def main():
    if args.seed is not None:
        setup_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.mode == 'constant':
        mode_name = f'constant-{args.strength}'
    else:
        mode_name = args.mode
    save_dir = os.path.join('results/aue', args.dataset, args.backbone, f'eps-{args.eps}',mode_name, f'seed-{args.seed}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log = logger(path=save_dir)
    log.info(str(args))


    dataloaders = PrepareDataLoaders(args.dataset, root=args.data, output_size=args.poison_size, for_gen=True,
                                     supervised=True, post_aug=args.post_aug, 
                                     strength=args.strength)
    train_loader = dataloaders.get_train_loader(args.batch_size, args.num_workers)
    test_loader = dataloaders.get_test_loader(args.batch_size, args.num_workers)

    if args.backbone == 'resnet18':
        model = resnet18(num_classes=args.num_classes).cuda()
        if args.dataset in ['cifar10', 'cifar100']:
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False).cuda()
            model.maxpool = nn.Identity().cuda()
    elif args.backbone == 'resnet50':
        if args.dataset in ['cifar10', 'cifar100']:
            model = resnet50(num_classes=args.num_classes).cuda()
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False).cuda()
        model.maxpool = nn.Identity().cuda()
    elif args.backbone == 'vgg19':
        model = vgg19(num_classes=args.num_classes).cuda()
    elif args.backbone == 'mobilenet':
        model = MobileNetV2(num_classes=args.num_classes).cuda()
    elif args.backbone == 'densenet121':
        model = DenseNet121(num_classes=args.num_classes).cuda()
    else:
        raise AssertionError('model is not defined')

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise AssertionError('optimizer is not defined')

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step,
                                                args.epochs,
                                                1,
                                                1e-6 / args.lr,
                                                warmup_steps=0)
    )

    if args.class_wise:
        poison = torch.zeros(args.num_classes, 3, args.poison_size, args.poison_size).cuda()
    else:
        poison = torch.zeros(len(train_loader.dataset), 3, args.poison_size, args.poison_size).cuda()

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim'])
        for i in range(start_epoch - 1):
            scheduler.step()
        log.info(f"RESUME FROM EPOCH {start_epoch-1}")
        poison = torch.load(os.path.join(save_dir, 'poison.pt'), map_location='cuda')

    for epoch in range(start_epoch, args.epochs + 1):
        if args.post_aug:
            aug = Augment(1.0, args.poison_size).aug_id
        else:
            current_strength = dynamic_strength(args.mode, epoch, args.epochs, args.dynamic_frequency)
            log.info(f'Epoch[{epoch}/{args.epochs}]\t'
                     f'current strength = {current_strength}')
            aug = Augment(current_strength, args.poison_size).aug_cl
        poison = train_epoch(train_loader, model, poison, optimizer, scheduler, epoch, log, aug)

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model.pt'))

            torch.save(poison, os.path.join(save_dir, 'poison.pt'))
            val_acc = evaluation(train_loader, model)
            test_acc = evaluation(test_loader, model)
            log.info(
                f'EVALUATION\t'
                f'val no-aug accuracy = {val_acc:.4f}\t'
                f'test accuracy = {test_acc:.4f}'
            )


if __name__ == '__main__':
    args = get_args()
    if args.class_wise:
        args.perturb_iters = 1

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.poison_size = 32
    if args.dataset == 'cifar100':
        args.num_classes = 100
        args.poison_size = 32
    if args.dataset == 'tinyimagenet':
        args.num_classes = 200
        args.poison_size = 64
    if args.dataset == 'miniimagenet':
        args.num_classes = 100
        args.poison_size = 84

    main()
