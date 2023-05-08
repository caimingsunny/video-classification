import time
import torch
# import os
# import sys
# import pdb
import torch.distributed as dist
import torch.nn as nn

from utils import AverageMeter, calculate_accuracy


def freeze_bn(model):
    print("Freezing Mean/Var of BatchNorm2D.")
    print("Freezing Weight/Bias of BatchNorm2D.")
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False,
                rpn=None,
                det_interval=2,
                nrois=10):
    print('train at epoch {}'.format(epoch))

    model.train()
    if rpn is not None:
        rpn.eval()
    else:
        freeze_bn(model)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    # if torch.cuda.is_available():
    #     device = torch.cuda.current_device()
    #     print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    # else:
    #     print("No GPU available, using CPU instead.")

    for i, (inputs, targets) in enumerate(data_loader):
        print('i', i)
        inputs = inputs.to(device, non_blocking=True)
        data_time.update(time.time() - end_time)
        targets = targets.to(device, non_blocking=True)
        print(0)
        if rpn is not None:
            '''
                There was an unexpected CUDNN_ERROR when len(rpn_inputs) is
                decrased.
            '''
            N, C, T, H, W = inputs.size()
            print('01')
            # print(inputs.size())
            if i == 0:
                max_N = N
            # sample frames for RPN
            print('02')
            sample = torch.arange(0, T, det_interval)
            print('03')
            rpn_inputs = inputs[:, :, sample].transpose(1, 2).contiguous()
            print('04')
            rpn_inputs = rpn_inputs.view(-1, C, H, W)
            print('05')
            if len(inputs) < max_N:
                print("Modified from {} to {}".format(len(inputs), max_N))
                while len(rpn_inputs) < max_N * (T // det_interval):
                    rpn_inputs = torch.cat((rpn_inputs, rpn_inputs[:(max_N-len(inputs))*(T//det_interval)]))
            with torch.no_grad():
                proposals = rpn(rpn_inputs)
            proposals = proposals.view(-1, T//det_interval, nrois, 4)
            print('06')
            if len(inputs) < max_N:
                proposals = proposals[:len(inputs)]
            # proposals_d = proposals.detach()
            # print('inputs', inputs)
            # print('proposals', proposals)
            print(1)
            outputs = model(inputs, proposals)
            print(2)
            # update to the largest batch_size
            max_N = max(N, max_N)
            print(3)
        else:
            outputs = model(inputs)
        
        print(4)
        # print(torch.cuda.is_available())
        # print(torch.cuda.device_count())
        # print(torch.cuda.current_device())
        
        loss = criterion(outputs, targets)
        print(5)
        # acc = calculate_accuracy(outputs, targets)

        print(6)
        # losses.update(loss.item(), inputs.size(0))
        print(7)
        # accuracies.update(acc, inputs.size(0))

        print(8)
        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()

        print(9)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print(10)
        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': current_lr
            })
        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                                                            i + 1,
                                                            len(data_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time,
                                                            loss=losses,
                                                            acc=accuracies))

    print(11)

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)
