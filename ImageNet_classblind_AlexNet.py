import torch
from pruning_methods import weight_prune_entireneuron_classblind_AlexNet as prune
from utils.utils import copy_weights
from utils.utils import get_dimensions
import torchvision.models as models
from models.AlexNet_model import alexnet

import argparse
import random
import shutil
import time
import warnings

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import os
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import psutil


def train(train_loader, model, criterion, optimizer, epoch):#, val_loader, pruningperc):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
         
        #print(model.conv1.weight.grad)
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % print_freq_tr == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5)) 

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #return top1.avg

            if i % print_freq_val == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
           
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

model= alexnet(pretrained=True)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from subprocess import check_output
def get_pid(name):
    return check_output(["pidof",name])


workers=8
epochs=10
start_epoch=0
batch_size=128
initial_lr=5e-4
momentum=0.9
weight_decay=1e-4
print_freq_tr=100
print_freq_val=10
pretrained= True 
resume=''
data ='../imagenet'

best_prec1 = 0

best_acc1 = 0
last_accuracy=0
total_epochs_sofar=0

model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), initial_lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

# optionally resume from a checkpoint
if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

cudnn.benchmark = True

# Data loading code
traindir = os.path.join(data, 'train')
valdir = os.path.join(data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir, transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  normalize,
                                 ]))

print("loaders enter")
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True, sampler=None)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize,
        ])),
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)

#pruning percentage increase every pruning iteration (delta)
delta=5
i=0
iteration=0

print("accuracy of the pretrained model is", 56.55)
acc1 = validate(val_loader, model, criterion)
print("measured accuracy before training and before compressing is", acc1)

while i<100:
    iteration += 1
    last_accuracy=0
    lr=initial_lr
    if i!=0:
        print("\n loading model \n")
        model= resnet(pretrained=True)
        model.cuda()
        model.load_state_dict(torch.load('../saved/AlexNetpruned'+str(eval('i'))+'.pkl'))
        optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
    if i==40:
       delta=3
    i=i+delta
    print("iteration number",iteration, "pruning percent", i)
    
    masks, names = prune(model,i,1)
    
    model.set_masks(masks)
    
    acc1 = validate(val_loader, model, criterion)
    print("accuracy before training and after compressing is", acc1)
    torch.save(model.state_dict(), '../saved/AlexNetpruned'+str(eval('i'))+'.pkl')
    last_accuracy=acc1
    

    for epoch in range(start_epoch, epochs):
        if epoch%5==0 and epoch!=0:
            lr = lr/10
            print("\n lr changed \n", lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion)
        print("accuracy of this epoch is", acc1, "while last accuracy is", last_accuracy)
        
        last_accuracy=acc1
                    
        print("saving model")
        
        torch.save(model.state_dict(), '../saved/AlexNetpruned'+str(eval('i'))+'.pkl')
                    
    total_epochs_sofar+= epochs
        
    if total_epochs_sofar>=90:
        delta=0
    elif total_epochs_sofar>=100:
        break