import argparse
from cProfile import label
import imp
import os
import time
import math
from os import path, makedirs
from pathlib import Path

import torch.nn.functional as F
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms
import copy
import torch.nn as nn
from simsiam.model_factory import SimSiam


from loader import CIFAR10N, CIFAR100N, DatasetGenerator
from utils import adjust_learning_rate, AverageMeter, ProgressMeter, save_checkpoint_mine, accuracy, load_checkpoint, \
    ThreeCropsTransform, FourCropsTransform
import torchvision.transforms as transforms

from losses import PUILoss

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', default='../datasets/cifar-10-batches-py', type=str, help='path to dataset directory')
parser.add_argument('--exp_dir', default='./save', type=str, help='path to experiment directory')
parser.add_argument('--dataset', default='cifar10', type=str, help='path to dataset',
                    choices=["cifar10", "cifar100", "clothing","animal"])
parser.add_argument('--noise_type', default='sym', type=str, help='noise type: sym or asym', choices=["sym", "asym"])
parser.add_argument('--r', type=float, default=0.5, help='noise level')
parser.add_argument('--trial', type=str, default='1', help='trial id')
parser.add_argument('--img_dim', default=32, type=int)
parser.add_argument('--name', default='exp', help='save to project/name')

parser.add_argument('--arch', default='resnet18', help='Inception resnet18 model name is used for training')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=550, help='number of training epochs')
parser.add_argument('--start_epochs', type=int, default=150, help='number of using pseudo-label epochs')
parser.add_argument('--start_pui_epochs', type=int, default=10, help='number of using pseudo-label epochs')

parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
parser.add_argument('--m', type=float, default=0.99, help='moving average of probbility outputs')
parser.add_argument('--tau', type=float, default=0.8, help='contrastive threshold (tau)')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--all', default=50000, type=int, help='t')
parser.add_argument('--t', default=0.2, type=float, help='t')

parser.add_argument('--lama', default=8.0, type=float, help='lambda for contrastive regularization term')
parser.add_argument('--lamb', default=2.0, type=float, help='lambda for contrastive regularization term')
parser.add_argument('--lamc', default=2.0, type=float, help='lambda for contrastive regularization term')
parser.add_argument('--type', default='gce', type=str, help='ce or gce loss', choices=["ce", "gce"])
parser.add_argument('--beta', default=0.6, type=float, help='gce parameter')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

parser.add_argument('--mix_up', default=1, type=int, help='use mixup or not.')
parser.add_argument('--fine_tuning', type=int, default=0, help='finetuning or not')
parser.add_argument('--model_dir', type=str, default="", help='model weights dir')

args = parser.parse_args()
import random

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.dataset == 'cifar10':
    args.nb_classes = 10
elif args.dataset == 'cifar100':
    args.nb_classes = 100

from randaugment import RandAugmentMC

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong_1 = self.strong(x)
        strong_2 = self.strong(x)
        return self.normalize(weak), self.normalize(strong_1), self.normalize(strong_2)
    
class GCE_loss(nn.Module):
    def __init__(self, q=0.8):
        super(GCE_loss, self).__init__()
        self.q = q

    def forward(self, outputs, targets):
        # print('====threshold======', threshold)
        pred = F.softmax(outputs, dim=1)
        pred_y = torch.sum(targets * pred, dim=1)
        pred_y = torch.clamp(pred_y, 1e-4)
        final_loss = torch.mean(((1.0 - pred_y ** self.q) / self.q), dim=0)

        return final_loss

class CE_loss(nn.Module):
    def __init__(self):
        super(CE_loss, self).__init__()
        self.delta = 1e-7

    def forward(self, outputs, targets):
        # targets = torch.zeros(targets.size(0), args.nb_classes).cuda().scatter_(1, targets.view(-1, 1), 1)
        pred = F.softmax(outputs, dim=1)
        pred_y = -torch.sum(targets * torch.log(pred+self.delta), dim=1)
        pred_y = torch.clamp(pred_y, 1e-4)
        final_loss = torch.mean(pred_y, dim=0)
        return final_loss

if args.type == 'ce':
    criterion = CE_loss()
else:
    criterion = GCE_loss(args.beta)

criterion_PUI = PUILoss(2.0)
criterion , criterion_PUI = criterion.cuda() , criterion_PUI.cuda()


def set_model(args):
    model = SimSiam(args.m, args)
    model.cuda()
    return model


def set_loader(args):
    if args.dataset == 'cifar10':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_cls_transformcon = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

        train_set = CIFAR10N(root=args.data_root,
                             transform=TransformFixMatch(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                             noise_type=args.noise_type,
                             r=args.r)


        test_data = datasets.CIFAR10(root=args.data_root, train=False, transform=test_transform, download=True)

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

        test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)
    elif args.dataset == 'cifar100':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276))
        ])

        train_cls_transformcon = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.267, 0.256, 0.276])])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.267, 0.256, 0.276])])

        train_set = CIFAR100N(root=args.data_root,
                              transform=TransformFixMatch(mean=[0.5071, 0.4865, 0.4409], std=[0.267, 0.256, 0.276]),
                              noise_type=args.noise_type,
                              r=args.r)

        test_data = datasets.CIFAR100(root=args.data_root, train=False, transform=test_transform, download=True)

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

        test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)
    return train_loader, test_loader


## Input interpolation functions
def mix_data(x, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(2, 2)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    lam = max(lam, 1 - lam)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, index, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    # if loss_a.item() < 0:
    #     print(pred)
    #     print(y_a)
    return lam * loss_a + (1 - lam) * loss_b


def train(train_loader, model, criterion,criterion_PUI, optimizer, epoch, args, change_index):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    

    model.train()
    end = time.time()
    
    outputs_item = torch.zeros(args.all,args.nb_classes).cuda()
    for i, (images, targets, _, index) in enumerate(train_loader):
        bsz = targets.size(0)
        change_index_batch = change_index[index]
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)##weak
            images[1] = images[1].cuda(args.gpu, non_blocking=True)##strong
            images[2] = images[2].cuda(args.gpu, non_blocking=True)##strong
            targets = targets.cuda(args.gpu, non_blocking=True)

        #Interpolated inputs Mixup操作
        if args.mix_up == 1:
            images[0], _, _ = mix_data(images[0])
            image, _, _ = mix_data(images[1])
            images[2], _, _ = mix_data(images[2])
        # compute output
        p1, z2, output = model(image, images[2], images[0])
        output_strong = model.forward_test(images[1])

        # avoid collapsing and gradient explosion
        p1 = torch.clamp(p1, 1e-4, 1.0 - 1e-4)
        z2 = torch.clamp(z2, 1e-4, 1.0 - 1e-4)

        contrast_1 = torch.matmul(p1, z2.t())  # B X B

        # <q,z> 
        contrast_1 = -contrast_1 * torch.zeros(bsz, bsz).fill_diagonal_(1).cuda() + (
            (1 - contrast_1).log()) * torch.ones(bsz, bsz).fill_diagonal_(0).cuda()
        contrast_logits = 2 + contrast_1
        soft_targets = torch.softmax(output, dim=1)
        contrast_mask = torch.matmul(soft_targets, soft_targets.t()).clone().detach()
        contrast_mask.fill_diagonal_(1)
        # 选择大于tau的对
        pos_mask = (contrast_mask >= args.tau).float()

        # label_mask = np.equal.outer(targets.cpu(), targets.cpu()).astype(int) * 0.2 + 1

        contrast_mask_1 = contrast_mask * pos_mask
        contrast_mask_1 = contrast_mask_1 / contrast_mask_1.sum(1, keepdim=True) #* torch.Tensor(label_mask).cuda()


        loss_ctr_1 = (contrast_logits * contrast_mask_1).sum(dim=1).mean(0)

        # 对称预测
        loss_ctr = loss_ctr_1 

        targets = targets.long().cuda()
        targets = torch.zeros(targets.size(0), args.nb_classes).cuda().scatter_(1, targets.view(-1, 1), 1)

        output_p = output
        start_epoch = args.start_epochs
        if epoch >= start_epoch:
            outputs_item[index] = soft_targets * targets
            if epoch >= start_epoch +1:
                ### 构造伪标签标签
                p_lable = torch.zeros(targets.size(0), args.nb_classes).cuda()
                _, p_index = soft_targets.topk(1, dim=1, largest=True)
                for j in range(targets.size(0)):
                    p_lable[j][p_index[j]] = soft_targets[j][p_index[j]]
                targets = change_index_batch * targets - (1 - change_index_batch) * p_lable
                output_p = change_index_batch * output + (1 - change_index_batch) * output_strong

        loss_ce = criterion(output_p, targets)

        if epoch >= args.start_pui_epochs:
            loss_pui = criterion_PUI(torch.softmax(output, dim=1), torch.softmax(output_strong, dim=1))

            loss = args.lama * loss_ctr + args.lamb * loss_ce + args.lamc * loss_pui

            if i == 1:
                print(" loss_ce=" + str(loss_ce.item()) + " loss_ctr=" + str(loss_ctr.item()) +" loss_pui=" + str(loss_pui.item()) +" loss=" + str(loss.item()))
        else:
            loss = args.lama * loss_ctr + args.lamb * loss_ce
            if i == 1:
                print(" loss_ce=" + str(loss_ce.item()) + " loss_ctr=" + str(loss_ctr.item()) +" loss=" + str(loss.item()))

        # compute gradient and do SGD step
        # print("compute gradient and do SGD step...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # print("measure elapsed time...")
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print("change_nums=" + str(args.all - change_index.sum()))
    return losses.avg ,outputs_item


def validation(test_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    acc = AverageMeter('Loss', ':.4e')

    model.eval()
    end = time.time()
    with torch.no_grad():
        
        for i, (images, targets) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
                # targets = targets.cuda(args.gpu, non_blocking=True)
            # compute output
            outputs = model.forward_test(images)
            acc2 = accuracy(outputs, targets, topk=(1,))

            # measure elapsed time
            acc.update(acc2[0].item(), images[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    return acc.avg


def increment_path(path, exist_ok=False, sep='', mkdir=True):
    """
    创建运行保存目录
    """
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# def get_change_index(args, outputs_item):
#     change_index = torch.ones(args.all, 1).cuda()
#     _, indices = outputs_item.topk(int(args.t * args.all), largest=False)
#     change_index[indices] = 0
#     return change_index

import numpy
# def get_change_index(args, outputs_item):
#     change_index = torch.zeros(args.all, 1).cuda()
#     if args.dataset == 'cifar10':
#         num = numpy.array([5000,5000,5000,5000,5000,5000,5000,5000,5000,5000])
#     elif args.dataset == 'cifar100':
#         num = numpy.ones(100)*500
#     # num = numpy.array([86152,88131,85788,50092,18976,82829,80437,80699,73318,88588,42312,87958,59663,75057])
#     # num = numpy.array([5466,4608,5091,4841,4981,4913,5322,4999,4970,4809])
#     # print(outputs_item)
#     # print(outputs_item.size())
#     for i in range(args.nb_classes):
#         _, indices = outputs_item[:,i].topk(int((1-args.t) * num[i]),dim=0)
#         change_index[indices] = 1
#     print(change_index.sum())
#     return change_index

def get_change_index(args, outputs_queue):
    
    change_index = torch.ones(args.all, 1).cuda()
    # outputs_queue = torch.cat((outputs_item_0,outputs_item_1,outputs_item_2),0)
    avg = outputs_queue.sum(dim=0).sum(dim=1)/3
    variance = ((outputs_queue.sum(dim=2)-avg)*(outputs_queue.sum(dim=2)-avg)).sum(dim=0)
    if args.dataset == 'cifar10':
        indices = [i for i in range(len(avg)) if avg[i]<0.5 or variance[i]>0.4]
    elif args.dataset == 'cifar100':
        indices = [i for i in range(len(avg)) if avg[i]<0.2 or variance[i]>0.4]
    change_index[indices] = 0
    return change_index

def main():
    print(vars(args))

    train_loader, test_loader = set_loader(args)

    model = set_model(args)

    if args.fine_tuning == 1:
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True

    start_epoch = 0

    save_dir = str(increment_path(
        Path('runs/train/' + str(args.r) + 'nosicrate_' + 'fine' + str(args.fine_tuning) + '_' + args.name + '/')))
    w = save_dir + '/weights'
    Path(w).mkdir(parents=True, exist_ok=True)  # make dir
    # routine
    best_acc = 0.0
    
    # outputs_item = torch.zeros(args.all).cuda()
    outputs_item = torch.zeros(args.all,args.nb_classes).cuda()
    # outputs_item_0 = torch.zeros(args.all,args.nb_classes).cuda()
    # outputs_item_1 = torch.zeros(args.all,args.nb_classes).cuda()
    # outputs_item_2 = torch.zeros(args.all,args.nb_classes).cuda()
    outputs_queue = torch.zeros(3,args.all,args.nb_classes).cuda()


    with open(save_dir + '/log.txt', 'a') as f:
        f.write(args.__str__() + "\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_optim = epoch

        adjust_learning_rate(optimizer, epoch_optim, args)
        print("Training...")

        # train for one epoch
        time0 = time.time()
        # if epoch%3 == 0:
        #     outputs_item_0 = outputs_item
        # elif epoch%3 == 1:
        #     outputs_item_1 = outputs_item
        # elif epoch%3 == 2:
        #     outputs_item_2 = outputs_item
        outputs_queue[epoch%3] = outputs_item
        # change_index = get_change_index(args, torch.unsqueeze(outputs_item_0,dim=0),torch.unsqueeze(outputs_item_1,dim=0),torch.unsqueeze(outputs_item_2,dim=0))
        change_index = get_change_index(args, outputs_queue)
        train_loss,outputs_item = train(train_loader, model, criterion,criterion_PUI, optimizer, epoch, args, change_index)
        print("Train \tEpoch:{}/{}\ttime: {}\tLoss: {}".format(epoch, args.epochs, time.time() - time0, train_loss))

        time1 = time.time()
        val_top1_acc = validation(test_loader, model, epoch, args)
        print("Test\tEpoch:{}/{}\t time: {}\tAcc: {}".format(epoch, args.epochs, time.time() - time1, val_top1_acc))
        if val_top1_acc > best_acc:
            weight_name = w + '/best.pt'
            save_checkpoint_mine(epoch, model, optimizer, val_top1_acc, weight_name, "保存模型")
        best_acc = max(best_acc, val_top1_acc)
        # if (epoch % 50) == 0:
        #     weight_name = w + '/epoch' + str(epoch) + '.pt'
        #     save_checkpoint_mine(epoch, model, optimizer, val_top1_acc, weight_name, "保存模型")
        with open(save_dir + '/log.txt', 'a') as f:
            f.write(
                'epoch: {}\t train_loss: {}\t val_top1_acc: {} time: {}\n'.format(
                    epoch, train_loss, val_top1_acc, time.time() - time0))
        # scheduler.step()
    print(
        'dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: gce \t beta:{}\t seed: {'
        '} \t best_acc: {}\tlast_acc: {}\n'.format(
            args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.beta, args.seed, best_acc,
            val_top1_acc))
    with open(save_dir + '/log.txt', 'a') as f:
        if args.type == 'ce':
            f.write(
                'dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: ce \t seed: {} \t '
                'best_acc: {}\tlast_acc: {}\n'.format(
                    args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.seed, best_acc, val_top1_acc))
        elif args.type == 'gce':
            f.write(
                'dataset: {}\t noise_type: {}\t noise_ratio: {} \tlamb: {}\t tau: {}\t type: gce \t beta:{}\t seed: {'
                '} \t best_acc: {}\tlast_acc: {}\n'.format(
                    args.dataset, args.noise_type, args.r, args.lamb, args.tau, args.beta, args.seed, best_acc,
                    val_top1_acc))



if __name__ == '__main__':
    main()
