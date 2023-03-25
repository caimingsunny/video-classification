import torch
import time
import sys
import pdb
import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, # 当前训练轮次
              data_loader, # 用于加载验证数据的数据加载器
              model, # 待验证的模型
              criterion, # 损失函数
              device, # 训练设备
              logger, # 记录日志信息的对象
              tb_writer=None, # 用于 TensorBoard 可视化的写入器
              distributed=False, # 是否使用分布式训练
              rpn=None, # 区域提议网络
              det_interval=2, # RPN 检测间隔
              nrois=10): # 每张图像中最大区域数量
    print('validation at epoch {}'.format(epoch))

    model.eval()
    if rpn is not None:
        rpn.eval()

    # 记录批处理时间、数据加载时间、损失和准确率
    batch_time = AverageMeter() #Computes and stores the average and current value
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    # 临时禁用梯度计算，以提高计算效率
    with torch.no_grad():
        # 循环遍历训练或验证数据集中的每个批次的数据和对应的目标标签
        for i, (inputs, targets) in enumerate(data_loader):
            # 记录每个批次数据的加载时间
            data_time.update(time.time() - end_time)
            targets = targets.to(device, non_blocking=True)
            # 判断是否使用了Region Proposal Network（RPN）进行检测
            if rpn is not None:
                '''
                    There was an unexpected CUDNN_ERROR when len(rpn_inputs) is
                    decrased.
                '''
                T = inputs.shape[2]
                N, C, T, H, W = inputs.size()
                if i == 0:
                    max_N = N
                # sample frames for RPN
                # 获取每个视频帧的特征输入，用于RPN网络进行检测
                # det_interval是指定每隔多少个视频帧采样一次，sample是采样出来的帧的序号，rpn_inputs是采样后的视频帧特征输入
                sample = torch.arange(0,T,det_interval)
                rpn_inputs = inputs[:,:,sample].transpose(1,2).contiguous()
                rpn_inputs = rpn_inputs.view(-1,C,H,W)
                # 判断当前批次大小是否小于前面批次大小的最大值max_N，如果是，则通过重复一部分视频帧特征，将rpn_inputs的大小扩充至与最大批次大小相同
                if len(inputs) < max_N:
                    print("Modified from {} to {}".format(len(inputs), max_N))
                    rpn_inputs = torch.cat((rpn_inputs, rpn_inputs[:(max_N-len(inputs))*(T//det_interval)]))
                # 得到提议框proposals，其中nrois是指定的每个视频帧的提议框数量，4是指定每个提议框的坐标个数
                with torch.no_grad():
                    proposals = rpn(rpn_inputs)
                proposals = proposals.view(-1,T//det_interval,nrois,4)
                # 将提议框作为输入数据，传入模型进行检测，并且如果当前批次大小小于最大批次大小，则只保留前len(inputs)
                if len(inputs) < max_N:
                    proposals = proposals[:len(inputs)]
                outputs = model(inputs, proposals.detach())
                # update to the largest batch_size
                max_N = max(N, max_N)
            else:
                outputs = model(inputs)

            # 计算当前的损失和准确率
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

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

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)

    return losses.avg
