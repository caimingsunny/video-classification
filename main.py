from pathlib import Path
import json
import random
import os
import pdb
import numpy as np
import wandb
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from opts import parse_opts
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from strg import STRG
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset import get_training_data, get_validation_data, get_inference_data
from utils import Logger, worker_init_fn, get_lr
from training import train_epoch
from validation import val_epoch
import inference

from rpn import RPN

# 将一个对象转换成JSON格式的字符串
def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def get_opt():

    # 实现了命令行参数的解析，使得可以在命令行中直接传入参数而不需要修改代码。
    # 通过 argparse 库，代码定义了许多命令行参数，比如训练数据所在的根目录、视频路径、注释路径等等。
    # 可以通过在命令行中指定这些参数的值来控制程序的行为
    # 例如： python main.py --root_path /path/to/root --video_path /path/to/videos --n_classes 400
    # 这个命令行将程序的根目录设置为 /path/to/root，视频路径设置为 /path/to/videos，将 n_classes 参数设置为 400。
    opt = parse_opts() # opts.py

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset) # mean.py
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]

    if opt.distributed:
        opt.dist_rank = 0 #int(os.environ["OMPI_COMM_WORLD_RANK"])

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    
    '''
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)
            '''

    return opt

# 加载模型
def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model

# 恢复训练
def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1]) # spatial_transforms.py
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)

# 创建训练模型所需的数据加载器、数据变换器、优化器和学习率调度器等
def get_train_utils(opt, model_parameters):
    # 根据不同的值选择不同的空间变换器
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []
    # 如果 train_crop 为 'random'，则使用 RandomResizedCrop 进行随机裁剪
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop( # spatial_transforms.py
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    # 如果为 'corner'，则使用 MultiScaleCornerCrop 进行多尺度角落裁剪
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales)) # spatial_transforms.py
    # 如果为 'center'，则使用 Resize 和 CenterCrop 进行中心裁剪
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter()) 
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.append(ScaleValue(opt.value_scale)) 
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform) 

    # 根据不同的值选择不同的时间变换器
    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride)) # temporal_transforms.py
    # 如果 train_t_crop 为 'random'，则使用 TemporalRandomCrop 进行随机时间裁剪
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    # 如果为 'center'，则使用 TemporalCenterCrop 进行中心时间裁剪
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform) 

    # 获取训练数据
    train_data = get_training_data(opt.video_path, opt.annotation_path, # dataset.py
                                   opt.dataset, opt.input_type, opt.file_type,
                                   spatial_transform, temporal_transform)
    # 根据是否启用分布式训练来创建相应的数据采样器
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)

    # 输入参数 opt 中的各种参数创建优化器和学习率调度器
    if opt.is_master_node:
        # 将模型训练过程中的指标记录到文件中
        train_logger = Logger(opt.result_path / 'train.log',  # utils.py
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(  # utils.py
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones)

    return (train_loader, train_sampler, train_logger, train_batch_logger,
            optimizer, scheduler)


def get_val_utils(opt):
    # 获取归一化方法
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    # 空间变换
    spatial_transform = [
        Resize(opt.sample_size), # 将图片调整到指定的大小 # spatial_transforms.py
        CenterCrop(opt.sample_size), # 中心裁剪
        ToTensor() # 转化为张量
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    # 时间变换
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride)) # 抽样 # temporal_transforms.py
    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples)) # 裁剪 # temporal_transforms.py
    temporal_transform = TemporalCompose(temporal_transform) # temporal_transforms.py
    
    val_data, collate_fn = get_validation_data(opt.video_path, # dataset.py
                                               opt.annotation_path, opt.dataset,
                                               opt.input_type, opt.file_type,
                                               spatial_transform,
                                               temporal_transform)
    # 分布式训练
    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False)
    else:
        val_sampler = None
    # 数据加载器
    val_loader = torch.utils.data.DataLoader(val_data,
#                                             batch_size=opt.batch_size,
                                             (opt.batch_size //
                                                         opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn, # utils.py
                                             collate_fn=collate_fn)

    # 如果是主节点
    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc'])
    else:
        val_logger = None

    return val_loader, val_logger


def get_inference_utils(opt):
    # 如果opt.inference_crop不是'center'或'nocrop'，则会引发一个断言错误
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    # 空间变换
    spatial_transform = [Resize(opt.sample_size)] # spatial_transforms.py
    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    # 时间变换
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride)) # temporal_transforms.py
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform) # temporal_transforms.py

    # 获取数据
    inference_data, collate_fn = get_inference_data( # dataset.py
        opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
        opt.file_type, opt.inference_subset, spatial_transform,
        temporal_transform)

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn, # utils.py
        collate_fn=collate_fn)

    return inference_loader, inference_data.class_names


# 将模型的参数和优化器状态保存到文件中
def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    # 是否使用了数据并行（即使用多个GPU进行训练）
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if index >= 0 and opt.device.type == 'cuda':
#        opt.device = torch.device(f'cuda:{index}')
        opt.device = torch.device('cuda:{}'.format(index))
        print(opt.device)

    # 确定是否在分布式环境下运行，初始化
    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    # 根据指定的模型类型生成模型
    model = generate_model(opt) # model.py

    # 加载预训练模型
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                      opt.n_finetune_classes, opt.strg) # model.py
    if opt.strg:
        # 构建整个视频动作识别模型
        model = STRG(model, nclass=opt.n_classes, nrois=opt.nrois) # strg.py
        rpn = RPN(nrois=opt.nrois) # rpn.py 候选框
        rpn = make_data_parallel(rpn, opt.distributed, opt.device) # model.py 用多个 GPU 进行训练
    else:
        rpn = None

    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)

    model = make_data_parallel(model, opt.distributed, opt.device) # model.py

#    if opt.pretrain_path:
#        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module) # model.py
#    else:
    parameters = model.parameters()

    if opt.is_master_node:
        print(model)

    # 定义损失函数为交叉熵损失函数
    criterion = CrossEntropyLoss().to(opt.device)

    # 加载training data
    if not opt.no_train:
        (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)
        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, opt.begin_epoch, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    # 如果开启 TensorBoard 并且是主节点，则准备 TensorBoard 相关的日志。
    # 如果是第一次运行，则从头开始记录日志，否则从上次记录的位置开始记录。
    if opt.tensorboard and opt.is_master_node:
        #from torch.utils.tensorboard import SummaryWriter
        from tensorboardX import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    # 如果开启了 W&B，则初始化 W&B 相关的配置。
    if opt.wandb:
        name = str(opt.result_path)
        wandb.init(
            project='strg',
            name=name,
            config=opt,
            dir= name,
#            resume=str(opt.resume_path) != '',
            sync_tensorboard=True)


    # 训练模型
    prev_val_loss = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.distributed:
                train_sampler.set_epoch(i)
            # 获取一个优化器中的学习率
            current_lr = get_lr(optimizer) # utils.py
            train_epoch(i, train_loader, model, criterion, optimizer, # training.py
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, opt.distributed,rpn=rpn,
                        det_interval=opt.det_interval, nrois=opt.nrois)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion, # validation.py
                                      opt.device, val_logger, tb_writer,
                                      opt.distributed, rpn=rpn,
                                    det_interval=opt.det_interval, nrois=opt.nrois)

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)

    if opt.inference:
        inference_loader, inference_class_names = get_inference_utils(opt)
        inference_result_path = opt.result_path / '{}.json'.format(
            opt.inference_subset)

        inference.inference(inference_loader, model, inference_result_path,
                            inference_class_names, opt.inference_no_average,
                            opt.output_topk) # inference.py


if __name__ == '__main__':
    opt = get_opt()

    # 初始化
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()

    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(-1, opt)
