import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import time
from utils import AverageMeter, cosine_annealing, logger, save_checkpoint, evaluation, setup_seed
from torch.nn import functional as F
from dataloaders import PrepareDataLoaders
from torchvision.models import resnet18, resnet50
from models import vgg19, MobileNetV2, DenseNet121
from torch import nn


def get_args():
    parser = argparse.ArgumentParser(description='EVAL-SL')
    parser.add_argument('--experiment', type=str, required=True, help='name of experiment')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg19',
                                                                             'mobilenet', 'densenet121'],
                        help='the model arch used in experiment')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                 'miniimagenet'],
                        help='the dataset used in experiment')
    parser.add_argument('--data', type=str, default='data', help='the directory of dataset')
    parser.add_argument('--num-classes', default=10, type=int, help='the number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)


    parser.add_argument('--poison-path', type=str, default=None, help='the path of pretrained poison')
    parser.add_argument('--poison-ratio', type=float, default=1.0, help='the poisoning ratio')
    parser.add_argument('--poison-size', type=int, default=32,
                        help='the image size of poisons')

    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='the optimizer used in training')
    parser.add_argument('--epochs', default=200, type=int,
                        help='the number of total epochs to run')
    parser.add_argument('--lr', default=0.5, type=float, help='optimizer learning rate')
    parser.add_argument('--seed', default=None, type=int, help='random seed')

    parser.add_argument('--resume', action='store_true', help='if resume training')
    parser.add_argument('--gpu-id', type=str, default='0', help='the gpu id')
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


def main():
    if args.seed is not None:
        setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    save_dir = os.path.join('results/eval', args.dataset, args.backbone, 'SL', args.experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log = logger(path=save_dir)
    log.info(str(args))

    if args.poison_path is not None:
        poison = torch.load(args.poison_path, map_location='cpu')
    else:
        poison = None


    dataloaders = PrepareDataLoaders(args.dataset, root=args.data, output_size=args.poison_size, for_gen=False,
                                    supervised=True, delta=poison, ratio=args.poison_ratio)
    train_loader = dataloaders.get_train_loader(args.batch_size, args.num_workers)
    test_loader = dataloaders.get_test_loader(args.batch_size, args.num_workers)

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
