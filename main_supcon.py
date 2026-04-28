from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900', help='where to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path', choices=['cifar10', 'cifar100', 'path'])
    parser.add_argument('--mean', type=str, help='mean of dataset (tuple)')
    parser.add_argument('--std', type=str, help='std of dataset (tuple)')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon', 'SimCLR'])
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain path (pth)')

    opt = parser.parse_args()

    if opt.dataset == 'path':
        assert opt.data_folder is not None and opt.mean is not None and opt.std is not None

    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = [int(it) for it in iterations]

    opt.model_name = '{}_{}_{}_lr_{}_bsz_{}_temp_{}_trial_{}'.format(
        opt.method, opt.dataset, opt.model, opt.learning_rate, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine: opt.model_name += '_cosine'
    if opt.batch_size > 256: opt.warm = True
    if opt.warm:
        opt.model_name += '_warm'
        opt.warmup_from, opt.warm_epochs = 0.01, 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder): os.makedirs(opt.tb_folder)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder): os.makedirs(opt.save_folder)

    return opt

def set_loader(opt):
    if opt.dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean, std = eval(opt.mean), eval(opt.std)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder, transform=TwoCropTransform(train_transform))
    else:
        ds_class = datasets.CIFAR10 if opt.dataset == 'cifar10' else datasets.CIFAR100
        train_dataset = ds_class(root=opt.data_folder, transform=TwoCropTransform(train_transform), download=True)

    return torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, 
                                       num_workers=opt.num_workers, pin_memory=True)

def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)
    if opt.syncBN: model = apex.parallel.convert_syncbn_model(model)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1: model.encoder = torch.nn.DataParallel(model.encoder)
        model, criterion = model.cuda(), criterion.cuda()
        cudnn.benchmark = True
    return model, criterion

def visualize_tsne(train_loader, model, device, opt, epoch):
    """Trích xuất feature và vẽ t-SNE"""
    model.eval()
    features, labels = [], []
    max_samples = 1500 # Giới hạn mẫu để chạy nhanh
    
    print(f"==> Đang tính toán t-SNE cho epoch {epoch}...")
    with torch.no_grad():
        for idx, (images, target) in enumerate(train_loader):
            img = images[0].to(device) # Lấy 1 view từ TwoCrop
            # Trích xuất từ encoder (không lấy projection head để cụm rõ hơn)
            feat = model.encoder(img)
            feat = torch.flatten(feat, 1)
            features.append(feat.cpu().numpy())
            labels.append(target.numpy())
            if len(np.concatenate(features)) >= max_samples: break

    features = np.concatenate(features)[:max_samples]
    labels = np.concatenate(labels)[:max_samples]
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeds = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeds[:, 0], embeds[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes", loc='upper right')
    plt.title(f't-SNE Visualization - Epoch {epoch}')
    
    save_path = os.path.join(opt.save_folder, f'tsne_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"==> Đã lưu biểu đồ t-SNE: {save_path}")
    model.train()

def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    batch_time, losses = AverageMeter(), AverageMeter()
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(features, labels) if opt.method == 'SupCon' else criterion(features)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0:
            print(f'Train: [{epoch}][{idx + 1}/{len(train_loader)}]\tBT {batch_time.avg:.3f}\tloss {losses.avg:.3f}')
    return losses.avg

def load_checkpoint(model, optimizer, save_file, device):
    print("==> Loading checkpoint...")
    checkpoint = torch.load(save_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'] + 1, checkpoint['opt']

def main():
    opt = parse_option()
    train_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    
    start_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opt.pretrain:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, opt.pretrain, device)

    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        print(f'epoch {epoch}, time {time.time() - time1:.2f}s, loss {loss:.4f}')

        logger.log_value('loss', loss, epoch)
        logger.log_value('lr', optimizer.param_groups[0]['lr'], epoch)

        # Lưu model và vẽ t-SNE định kỳ
        if epoch % opt.save_freq == 0:
            visualize_tsne(train_loader, model, device, opt, epoch)
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    save_model(model, optimizer, opt, opt.epochs, os.path.join(opt.save_folder, 'last.pth'))

if __name__ == '__main__':
    main()