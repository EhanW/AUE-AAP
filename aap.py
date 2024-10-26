import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import time
from utils import AverageMeter, cosine_annealing, logger, save_checkpoint, evaluation, setup_seed, Augment
from torch.nn import functional as F
from dataloaders import APDataLoaders
from torchvision.models import resnet18, resnet50
from models import vgg19, MobileNetV2, DenseNet121
from torch import nn



def get_args():
    parser = argparse.ArgumentParser(description='GEN-AAP')
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
    parser.add_argument('--poison-size', type=int, default=32,
                        help='the image size of poisons')
    parser.add_argument('--label-translation', type=int, default=1, 
                        help='the label traslation from the original class to the target class')
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='the optimizer used in training')
    parser.add_argument('--epochs', default=40, type=int,
                        help='the number of total epochs to run')
    parser.add_argument('--lr', default=0.5, type=float, 
                        help='optimizer learning rate')
    parser.add_argument('--seed', default=1, type=int, 
                        help='random seed')
    parser.add_argument('--untargeted', action='store_true', 
                        help='if untargeted AAP attack. default is targeted AAP attack')
    parser.add_argument('--ref-mode', default='augmented', type=str, choices=['standard', 'augmented'],
                        help='the augmentation mode in the update of reference model')
    parser.add_argument('--gen-mode', default='augmented', type=str, choices=['standard', 'augmented'],
                        help='the augmentation mode in the update of poisons')
    parser.add_argument('--ref-strength', default=1.0, type=float, 
                        help='the augmentation strength of reference model if ref-mode is augmented')
    parser.add_argument('--gen-strength', default=1.0, type=float, 
                        help='the augmentation strength of poisons if gen-mode is augmented')
    parser.add_argument('--perturb-iters', default=250, type=int,
                        help='the number of iterations in the PGD update of poisons')
    parser.add_argument('--resume', action='store_true', 
                        help='if resume training')
    parser.add_argument('--gpu-id', type=str, default='0', 
                        help='the gpu id')
    parser.add_argument('--eps', type=int, default=8, 
                        help='the L-inf norm constraint for perturbations')
    return parser.parse_args()


def train_epoch(train_loader, model, optimizer, scheduler, epoch, log):
    losses = AverageMeter()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    start = time.time()
    for i, (data, target, _) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        data_time = time.time() - start
        data_time_meter.update(data_time)

        features = model.train()(data)
        loss = F.cross_entropy(features, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.shape[0])

        train_time = time.time() - start
        train_time_meter.update(train_time)
        start = time.time()
    log.info(
        f'TRAINING\t'
        f'Epoch[{epoch}/{args.epochs}]\t'
        f'avg loss = {losses.avg:.4f}\t'
        f'epoch time = {train_time_meter.sum:.2f}\t'
        f'data time = {data_time_meter.sum:.2f}\t'
        f'current lr = {current_lr:.4f}'
    )
    scheduler.step()


def error_opt(model, data, target, index, poison, aug):
    eps = args.eps/255
    iters = args.perturb_iters
    alpha = eps*2.5/iters

    if not args.untargeted:
        target = (target+args.label_translation)%args.num_classes
    delta = poison[index]
    delta = torch.nn.Parameter(delta)

    for _ in range(iters):
        inputs = data + delta
        img = aug(inputs)
        features = model.eval()(img)
        model.zero_grad()
        loss = F.cross_entropy(features, target)
        loss.backward()
        if args.untargeted:
            delta.data = delta.data + alpha * delta.grad.sign()
        else:
            delta.data = delta.data - alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(data + delta.data, min=0, max=1) - data
        
    return delta.detach()


def perturb(train_loader, model, poison, aug, log, num_perturbs=10000):
    model.requires_grad_(False)
    gen_time_meter = AverageMeter()
    start = time.time()
    for i, (data, target, index) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        poison[index] = error_opt(model, data, target, index, poison, aug)

        gen_time = time.time() - start
        gen_time_meter.update(gen_time)
        start = time.time()
        log.info(
            f'GENERATING\t'
            f'Batch[{i}/{len(train_loader)}]\t'
            f'batch time: {gen_time}'
        )
        if (i+1) % num_perturbs == 0:
            break
    model.requires_grad_(True)
    log.info(
        f'Total Generation Time: {gen_time_meter.sum}'
    )
    return poison


def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.ref_mode == 'standard':
        name1 = 'standard'
    else:
        name1 = str(args.ref_strength)
    if args.gen_mode == 'standard':
        name2 = 'standard'
    else:
        name2 = str(args.gen_strength)
    name = name1 + '_' + name2

    if args.seed is not None:
        setup_seed(args.seed)
        name = name

    if args.untargeted:
        mode = 'untargeted'
    else:
        mode = 'targeted'

    save_dir = os.path.join('results/aap', args.dataset, args.backbone, f'eps-{args.eps}', mode, str(args.epochs), name, str(args.label_translation), f'seed{args.seed}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log = logger(path=save_dir)
    log.info(str(args))

    dataloaders = APDataLoaders(args.dataset, args.data, args.poison_size, ref_mode=args.ref_mode, rrc=args.ref_strength, cj=args.ref_strength, rg=args.ref_strength)
    train_loader = dataloaders.get_train_loader(args.batch_size, args.num_workers)
    test_loader = dataloaders.get_test_loader(args.batch_size, args.num_workers)
    gen_loader = dataloaders.get_val_loader(args.batch_size, args.num_workers)

    if args.backbone == 'resnet18':
        model = resnet18(num_classes=args.num_classes).cuda()
        if args.dataset in ['cifar10', 'cifar100']:
            model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False).cuda()
            model.maxpool = nn.Identity().cuda()
    elif args.backbone == 'resnet50':
        model = resnet50(num_classes=args.num_classes).cuda()
        if args.dataset in ['cifar10', 'cifar100']:
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
                                                warmup_steps=10)
    )

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(os.path.join(save_dir, 'model.pt'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim'])
        for i in range(start_epoch - 1):
            scheduler.step()
        log.info(f"RESUME FROM EPOCH {start_epoch-1}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch, log)

        if epoch % 25 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model.pt'))

            val_acc = evaluation(train_loader, model)
            test_acc = evaluation(test_loader, model)
            log.info(
                f'val accuracy = {val_acc:.4f}\t'
                f'test accuracy = {test_acc:.4f}'
            )

    
    poison = torch.zeros(len(gen_loader.dataset), 3, args.poison_size, args.poison_size).cuda()
    if args.gen_mode == 'standard':
        aug = Augment(args.gen_strength, args.poison_size).aug_standard
    elif args.gen_strength > 0:
        aug = Augment(args.gen_strength, args.poison_size).aug_cl
    else:
        aug = Augment(args.gen_strength, args.poison_size).aug_id
    poison = perturb(gen_loader, model, poison, aug, log=log)
    torch.save(poison, os.path.join(save_dir, 'poison.pt'))


if __name__ == '__main__':
    args = get_args()
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
    cudnn.benchmark = True
    main()
