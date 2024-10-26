import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import time
from utils import AverageMeter, nt_xent, cosine_annealing, logger, save_checkpoint, LARS, evaluation, byol_mse, setup_seed
from torch.nn import functional as F
from models import BYOL, SimCLR, MoCo, LinearClassifier, SimSiam
from torch import nn
from dataloaders import PrepareDataLoaders


def get_args():
    parser = argparse.ArgumentParser(description='EVAL-CL')
    parser.add_argument('--experiment', type=str, required=True, help='name of experiment')
    parser.add_argument('--method', default='SimCLR', choices=['SimCLR', 'BYOL', 'MoCo', 'SimSiam'],
                        help='the contrastive algorithm used in training')

    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg19',
                                                                             'mobilenet', 'densenet121'],
                        help='the model arch used as backbone')
    parser.add_argument('--projection-dim', default=128, type=int, help='the projection dimensions')
    parser.add_argument('--num-classes', default=10, type=int, help='the number of classes in the dataset')
    parser.add_argument('--poison-path', type=str, default=None, help='the path of pretrained poison')
    parser.add_argument('--poison-ratio', type=float, default=1.0, help='the poisoning ratio')
    parser.add_argument('--poison-size', type=int, default=32, help='the image size of poisons')

    parser.add_argument('--save-frequency', default=100, type=int, help='saving frequency of checkpoints')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                 'miniimagenet'],
                        help='the dataset used in experiment')
    parser.add_argument('--data', type=str, default='data', help='the directory of dataset')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--epochs', default=1000, type=int,
                        help='the number of total epochs to run')
    parser.add_argument('--lr', default=0.5, type=float, help='optimizer learning rate')
    parser.add_argument('--temperature', default=0.5, type=float, help='the temperature in SimCLR and MoCo')
    parser.add_argument('--encoder-momentum', default=0.99, type=float, help='the encoder momentum in MoCo and BYOL')
    parser.add_argument('--queue-length', default=4096, type=int, help='queue length in MoCo')
    parser.add_argument('--resume', action='store_true', help='if resume training')
    parser.add_argument('--gpu-id', type=str, default='0', help='the gpu id')
    parser.add_argument('--linear-probe-lr', type=float, default=1.0, help='the learning rate of linear probing')
    parser.add_argument('--seed', default=None, type=int, help='random seed')

    return parser.parse_args()


def pretraining_epoch(train_loader, model, optimizer, scheduler, epoch, log):
    losses = AverageMeter()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    start = time.time()
    for i, (data, _, _) in enumerate(train_loader):
        data = data.cuda()
        shape = data.shape
        data = data.reshape(shape[0]*2, shape[2], shape[3], shape[4])
        data_time = time.time() - start
        data_time_meter.update(data_time)
        if args.method == 'SimCLR':
            features = model.train()(data)
            loss = nt_xent(features, t=args.temperature)
        elif args.method == 'BYOL':
            model.momentum_update_target_encoder()
            online_output, target_output = model.train()(data)
            loss = byol_mse(online_output, target_output)
        elif args.method == 'MoCo':
            model.momentum_update_key_encoder()
            features, labels = model.train()(data)
            loss = F.cross_entropy(features, labels)
        elif args.method == 'SimSiam':
            loss = model.train()(data[::2], data[1::2])
        else:
            raise AssertionError('contrastive algorithm is not defined')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.shape[0])

        train_time = time.time() - start
        train_time_meter.update(train_time)
        start = time.time()
    log.info(
        f'PRETRAINING\t'
        f'Epoch[{epoch}/{args.epochs}]\t'
        f'avg loss = {losses.avg:.4f}\t'
        f'epoch time = {train_time_meter.sum:.2f}\t'
        f'data time = {data_time_meter.sum:.2f}\t'
        f'current lr = {current_lr:.4f}'
    )
    scheduler.step()


def linear_probing(val_loader, test_loader, model, log, num_classes, save_dir):
    if args.method == 'SimCLR':
        encoder = model.enc
    elif args.method == 'BYOL':
        encoder = model.online_encoder
    elif args.method == 'MoCo':
        encoder = model.encoder_q
    elif args.method == 'SimSiam':
        encoder = model.enc
    else:
        raise AssertionError('contrastive algorithm is not defined')

    if args.backbone == 'resnet18':
        feature_dim = 512
    elif args.backbone == 'resnet50':
        feature_dim = 2048
    elif args.backbone == 'vgg19':
        feature_dim = 512
    elif args.backbone == 'densenet121':
        feature_dim = 1024
    elif args.backbone == 'mobilenet':
        feature_dim = 1280
    else:
        raise AssertionError('model is not defined')

    encoder.fc = nn.Identity()
    cls = LinearClassifier(encoder, feature_dim, num_classes).cuda()
    cls.enc.requires_grad_(False)
    optimizer = torch.optim.SGD(cls.lin.parameters(), lr=args.linear_probe_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(60, 75, 90), gamma=0.2)

    for epoch in range(100):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        losses = AverageMeter()
        for i, (data, target, _) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            loss = F.cross_entropy(cls.eval()(data), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), data.shape[0])
        log.info(f'LINEAR PROBING\t'
                 f'Epoch[{epoch+1}/100]\t'
                 f'avg loss = {losses.avg:.4f}\t'
                 f'current lr = {current_lr:.4f}')
        scheduler.step()

        if (epoch+1) % 50 == 0:
            val_acc = evaluation(val_loader, cls)
            test_acc = evaluation(test_loader, cls)
            log.info(
                f'val accuracy = {val_acc:.4f}\t'
                f'test accuracy = {test_acc:.4f}'
            )
    torch.save(cls.state_dict(), os.path.join(save_dir, 'cls.pt'))


def main():
    if args.seed is not None:
        setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    save_dir = os.path.join('results/eval', args.dataset, args.backbone, args.method, args.experiment)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log = logger(path=save_dir)
    log.info(str(args))

    if args.poison_path is not None:
        poison = torch.load(args.poison_path, map_location='cpu')
    else:
        poison = None

    dataloaders = PrepareDataLoaders(args.dataset, root=args.data, output_size=args.poison_size, for_gen=False,
                                    supervised=False, delta=poison, ratio=args.poison_ratio)
    train_loader = dataloaders.get_train_loader(args.batch_size, args.num_workers)
    val_loader = dataloaders.get_val_loader(args.batch_size, args.num_workers)
    test_loader = dataloaders.get_test_loader(args.batch_size, args.num_workers)
    if args.dataset in ['cifar10', 'cifar100']:
        cifar_conv = True
    else:
        cifar_conv = False
    if args.method == 'SimCLR':
        model = SimCLR(backbone=args.backbone, projection_dim=args.projection_dim, cifar_conv=cifar_conv).cuda()
    elif args.method == 'BYOL':
        model = BYOL(backbone=args.backbone, projection_dim=args.projection_dim, m=args.encoder_momentum, cifar_conv=cifar_conv).cuda()
    elif args.method == 'MoCo':
        model = MoCo(backbone=args.backbone, projection_dim=args.projection_dim, K=args.queue_length,
                     T=args.temperature, m=args.encoder_momentum, cifar_conv=cifar_conv).cuda()
    elif args.method == 'SimSiam':
        model = SimSiam(backbone=args.backbone, cifar_conv=cifar_conv).cuda()
    else:
        raise AssertionError('contrastive algorithm is not defined')

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
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
        pretraining_epoch(train_loader, model, optimizer, scheduler, epoch, log)

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, 'model.pt'))

        if epoch % args.save_frequency == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, filename=os.path.join(save_dir, f'model_{epoch}.pt'))

    linear_probing(val_loader, test_loader, model, log, num_classes=args.num_classes, save_dir=save_dir)


if __name__ == '__main__':
    args = get_args()

    if args.method == 'SimCLR':
        args.lr = 0.5
        args.temperature = 0.5
    if args.method == 'BYOL':
        args.lr = 1.0
        args.encoder_momentum = 0.999
    if args.method == 'MoCo':
        args.lr = 0.3
        args.temperature = 0.2
        args.encoder_momentum = 0.99
    if args.method == 'SimSiam':
        args.lr = 0.1
        args.linear_probe_lr = 5.0

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
