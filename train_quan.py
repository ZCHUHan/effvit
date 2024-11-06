from ast import arg
from asyncio import base_tasks
from calendar import c
from json import load
from math import e
from multiprocessing import reduction
from sympy import E
from tensorboard import summary
from tomlkit import date
from tqdm import tqdm
import logging
from models.efficientvit import cls
import network
import utils
import os
import random
import argparse
import numpy as np
import math
import warnings
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import torch.utils.data as data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter 



from datasets import VOCSegmentation, Cityscapes
from metrics import StreamSegMetrics
from models.seg_model_zoo import create_seg_model
from models.utils import resize
from models.nn.quant_lsq import QuanConv, PActFn, PACT, SymmetricQuantFunction
# from utils.loss import ChannelWiseDivergence as CWD_loss

from utils import ext_transforms as et
from utils.visualizer import Visualizer
from utils.loss import KLLossSoft as KL_loss
from utils.loss import CriterionCWD as CWD_loss
from utils.loss import MultiClassDiceLoss as Dice_Loss
from utils.scheduler import CosineAnnealingWarmupRestarts as CosineLR



def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    # parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        # choices=available_models, help='model name')
    parser.add_argument("--model", type=str)
    parser.add_argument("--quan_model", type=str)
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=100e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step', 'cosine'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size_h", type=int, default=960)
    parser.add_argument("--crop_size_w", type=int, default=1920)
    # ddp
    parser.add_argument("--worker", type=int, default=2)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    # ckpt
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt_teacher", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt_saved", default='default', type=str,
                        help="save checkpoint to")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'kd_pixel', 'kd_channel'], help="loss type (default: False)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    # effvit options
    parser.add_argument("--log_dir", type=str, default='./log')
    parser.add_argument("--weight_url", type=str)
    parser.add_argument("--weight_url_teacher", type=str)
    return parser

def main():
    args = get_argparser().parse_args()
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print('ngpus_per_node', ngpus_per_node)
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # Setup visualization
    if args.dataset.lower() == 'voc':
        args.num_classes = 21
    elif args.dataset.lower() == 'cityscapes':
        args.num_classes = 19
   
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu # local rank
        print(args.dist_url)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.rank in [-1, 0]:
        summary_writer = SummaryWriter(args.log_dir)
    
    vis = Visualizer(port=args.vis_port,
                     env=args.vis_env) if args.enable_vis else None

    if vis is not None:  # display options
        vis.vis_table("Options", vars(args))

    # setup seeds 
    torch.manual_seed(args.random_seed + (int(args.gpu_id) if args.gpu is None else args.gpu))
    np.random.seed(args.random_seed + (int(args.gpu_id) if args.gpu is None else args.gpu))
    random.seed(args.random_seed + (int(args.gpu_id) if args.gpu is None else args.gpu))

    # setup dataset and dataloader
    train_dst, val_dst = get_dataset(args)
    if args.distributed:
        if args.gpu is not None:
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.worker = int((args.worker + ngpus_per_node - 1) / ngpus_per_node)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)
    else:
        train_sampler = None
    train_loader = data.DataLoader(
        train_dst, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.worker,
        pin_memory=True, sampler=train_sampler)  # drop_last=True to ignore single-image batches.
    
    val_loader = data.DataLoader(
        val_dst, batch_size=1, shuffle=True, num_workers=args.worker,
        pin_memory=True)
    # create model
    if args.rank in [-1, 0]:
        print(f"Creating student model: {args.model}")
        print(f"Creating teacher model: {args.quan_model}")

    model_teacher = create_seg_model(args.model, args.dataset, weight_url=args.ckpt_teacher)
    for p in model_teacher.parameters():
        p.requires_grad = False
    model = create_seg_model(args.quan_model, args.dataset, weight_url=args.ckpt)
    # load pth
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        if args.rank in [-1,0]:
            print("=> loading checkpoint for student model'{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        try:
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as e:
            if args.rank in [-1, 0]:
                print(e)
    else:
        if args.rank in [-1, 0]:
            print("[!] Retrain Student Model from scratch")
    
    
    if args.ckpt_teacher is not None and os.path.isfile(args.ckpt_teacher):
        if args.rank in [-1, 0]:
            print("=> loading checkpoint for student model'{}'".format(args.ckpt_teacher))
        checkpoint_teacher = torch.load(args.ckpt_teacher, map_location=torch.device('cpu'))
        if "state_dict" in checkpoint_teacher:
            checkpoint_teacher = checkpoint_teacher["state_dict"]
        else:
            checkpoint_teacher = checkpoint_teacher["model_state"]
        try:
            model_teacher.load_state_dict(checkpoint_teacher)    
        except RuntimeError as e:
            if args.rank in [-1, 0]:
                print(e)

    # initialize quantization scale
    if args.rank in [-1, 0]:
        print("Initializing quantization scale")
        model.eval()
        with torch.no_grad():
            inputs, _ = next(iter(train_loader))
            inputs = inputs.to(device="cuda:{}".format(str(args.gpu)), dtype=torch.float32)
            print("inputs shape: ", inputs.shape)
            _ = model(inputs)
        print("Initializing quantization scale done")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        print("Distributed training")
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = 'cuda'+':'+str(args.gpu)
            model.cuda(args.gpu)
            model_teacher.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            # model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu])
        else:
            raise Exception("args.gpu is None")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_teacher = model_teacher.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print("using DataParallel")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)
        print("CUDA_VISIBLE_DEVICES: %s" % os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model).to(device)
        model_teacher = torch.nn.DataParallel(model_teacher).to(device)
    model_teacher.eval() 
    optimizer = torch.optim.AdamW(params=model.parameters(),
                             lr=args.lr,
                             betas=(0.9, 0.999),
                             weight_decay=args.weight_decay)   
    # print('activation scale {}'.format(model.module.backbone.input_stem[0].conv.lsq_a.s))
    # print('activation scale {}'.format(model.module.backbone.input_stem[0].conv.lsq_w.s))
    # print('syncbatchnorm? :{}'.format(isinstance(model.module.backbone.input_stem[1].depth_conv.conv.norm, torch.nn.SyncBatchNorm)))
    # return
    if args.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif args.loss_type == 'kd_channel':
        # KD_loss = CWD_loss(tau=4.0, loss_weight=3.0, size_average=True) 
        KD_loss = CWD_loss(temperature=4.0, loss_weight=3.0) 
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif args.loss_type == 'kd_pixel':
        KD_loss = KL_loss()
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        # criterion = Dice_Loss()
    if not args.test_only:
        KD_loss = KD_loss.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
    
    def save_ckpt(args, path):
        """ save current model
        """
        if args.distributed and args.rank in [-1, 0]:
            torch.save({
                "cur_itrs": cur_itrs,
                "cur_epochs": cur_epochs,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
            }, path)
            if args.rank == 0:
                print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
          
    if args.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, args.total_itrs, power=1)
    elif args.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    elif args.lr_policy == 'cosine':
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.total_itrs)
        scheduler = CosineLR(optimizer, first_cycle_steps=50000, max_lr=args.lr, min_lr=6e-7, warmup_steps=1000)
     
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    # resume training
    if args.continue_training:
        if os.path.isfile(args.ckpt):
            if args.gpu in [-1, 0]:
               print("[!] Load Checkpoint %s" % args.ckpt)
            if args.gpu is None:
                checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.ckpt, map_location=loc)
            if args.gpu in [-1, 0]:
               print("[!] Continue training")
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            cur_epochs = checkpoint["cur_epochs"]
            best_score = checkpoint['best_score']
            scheduler.step(cur_itrs)
            if args.gpu in [-1, 0]:
               print("Training state restored from %s" % args.ckpt)
    cudnn.benchmark = True

    metrics = StreamSegMetrics(args.num_classes)
    vis_sample_id = np.random.randint(0, len(val_loader), args.vis_num_samples,
                                      np.int32) if args.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
    
    if args.test_only and args.rank in [-1, 0]:
        model.eval()
        val_score, ret_samples = validate(
            args=args, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        
        return
    

    interval_loss = 0
    interval_loss_seg = 0
    interval_loss_kd = 0
    interval_loss_psum = 0
    avg_iter_duration = timedelta(0)

    if args.rank in [-1, 0]:
        model.eval()
        val_score, ret_samples = validate(
            args=args, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        val_score_t, ret_samples_t = validate(
                args=args, model=model_teacher,
                loader=val_loader, device=device,
                metrics=metrics,ret_samples_ids=vis_sample_id)
        print("teacher model scores:\n", metrics.to_str(val_score_t))
        
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        if args.distributed:
            train_sampler.set_epoch(cur_epochs)
        cur_epochs += 1
        
        model.train()
        for (images, labels) in train_loader:
            cur_itrs += 1
            iter_start_time = datetime.now()
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            #     images = images.to(dtype=torch.float32)

            # if torch.cuda.is_available():
            #     labels = labels.cuda(args.gpu, non_blocking=True)
            #     labels = labels.to(dtype=torch.long)

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images, False)
            if args.ckpt_teacher is not None:
                outputs_teacher = model_teacher(images)

            if outputs.shape[2:] != labels.shape[-2:]:
                outputs = resize(outputs, size=labels.shape[-2:])
                # print(outputs_teacher)
                if args.ckpt_teacher is not None:
                    outputs_teacher = resize(outputs_teacher, size=labels.shape[-2:])

            loss_seg = criterion(outputs, labels)
            if args.ckpt_teacher is not None:
                loss_kd = KD_loss(outputs, outputs_teacher)
            # loss_psum = loss_psum.mean()
            # loss_kd = KD_loss(outputs_conv, outputs_teacher_conv)
            # loss = loss_kd / loss_kd.detach() + loss_seg / loss_seg.detach()#+ loss_psum
            if args.ckpt_teacher is not None:
                loss = loss_kd + loss_seg # + loss_psum
            else: loss = loss_seg
            # print("loss_seg: ", loss_seg)
            # print("loss_psum: ", loss_psum)
            loss.backward()
            optimizer.step()
            
            iter_end_time = datetime.now()
            iter_duration = iter_end_time - iter_start_time
            avg_iter_duration = (avg_iter_duration * (cur_itrs - 1) + iter_duration) / cur_itrs


            remaining_iters = args.total_itrs - cur_itrs
            estimated_remaining_time = avg_iter_duration * remaining_iters


            estimated_remaining_seconds = estimated_remaining_time.total_seconds()
            estimated_remaining_hours = estimated_remaining_seconds / 3600

            
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            np_loss_seg = loss_seg.detach().cpu().numpy()
            interval_loss_seg += np_loss_seg

            np_loss_kd = loss_kd.detach().cpu().numpy()
            interval_loss_kd += np_loss_kd
            
            # np_loss_psum = reduce_value(loss_psum, average=True).detach().cpu().numpy()
            # interval_loss_psum += np_loss_psum
            
            torch.cuda.synchronize()
            if (cur_itrs) % 50 == 0 and args.rank in [-1, 0]:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                current_lr = scheduler.get_lr()[0]
                # current_lr = scheduler.get_last_lr()[0]
                interval_loss = interval_loss / 50
                interval_loss_seg = interval_loss_seg / 50
                interval_loss_kd = interval_loss_kd / 50
                # interval_loss_psum = interval_loss_psum / 50
                summary_writer.add_scalar('Loss/train', interval_loss, cur_itrs)
                summary_writer.add_scalar('Loss_seg/train', interval_loss_seg, cur_itrs)
                summary_writer.add_scalar('Loss_kd/train', interval_loss_kd, cur_itrs)
                # summary_writer.add_scalar('Loss_psum/train', interval_loss_psum, cur_itrs)
                summary_writer.add_scalar('LR', current_lr, cur_itrs)
                print(f"[{current_time}] "
                      f"Epoch {cur_epochs}, Itrs {cur_itrs}/{int(args.total_itrs)}, "
                      f"Loss={interval_loss:.4f}, Loss_seg={interval_loss_seg:.4f}, " #Loss_psum={interval_loss_psum:.4f},"
                      f"Loss_kd={interval_loss_kd:.4f}, LR={current_lr:.5e}, "
                      f"Estimated Remaining Time: {estimated_remaining_hours:.2f}h")
                # print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    #   (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0
                interval_loss_seg = 0.0
                interval_loss_kd = 0.0
                # interval_loss_psum = 0.0

            if (cur_itrs) % args.val_interval == 0 and args.rank in [-1, 0]:
                # save_ckpt(args, 'checkpoints/%s/%s_%s_%s_os%d.pth' %
                #           (args.ckpt_saved, cur_itrs, args.model, args.dataset, args.output_stride))
                save_ckpt(args, 'checkpoints/%s/latest_%s_%s_os%d.pth' %
                          (args.ckpt_saved, args.model, args.dataset, args.output_stride))
                print("validation...")

                val_score, ret_samples = validate(
                    args=args, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print("teacher model scores:\n", metrics.to_str(val_score_t))
                print("student model scores:\n", metrics.to_str(val_score))

                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(args, 'checkpoints/%s/best_%s_%s_os%d.pth' %
                              (args.ckpt_saved, args.model, args.dataset, args.output_stride))
                summary_writer.add_scalar('mIoU_student/val', val_score['Mean IoU'], cur_itrs)
                summary_writer.add_scalar('Acc_student/val', val_score['Overall Acc'], cur_itrs)

                summary_writer.add_scalar('mIoU_teacher/val', val_score_t['Mean IoU'], cur_itrs)
                summary_writer.add_scalar('Acc_teacher/val',  val_score_t['Overall Acc'], cur_itrs)

                # if vis is not None:  # visualize validation score and samples
                #     vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                #     vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                #     vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                #     for k, (img, target, lbl) in enumerate(ret_samples):
                #         img = (denorm(img) * 255).astype(np.uint8)
                #         target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                #         lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                #         concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                #         vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

        if cur_itrs >= args.total_itrs:
            return
    

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size_h, opts.crop_size_w), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size_h),
                et.ExtCenterCrop(opts.crop_size_w),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size_h, opts.crop_size_w)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size_h, opts.crop_size_w)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst



        
def validate(args, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    model.eval()
    metrics.reset()
    ret_samples = []
    if args.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0
        
    # if args.rank in [-1, 0]:  # 只在 rank 为 0 的进程上创建 tqdm 进度条
        # loader = tqdm(loader)
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            # print("images shape: ", images.shape)
            
            images = images.to(device, dtype=torch.float32)   
            labels = labels.to(device, dtype=torch.long)
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            #     images = images.to(dtype=torch.float32)
            # else:
            #     images = images.to(device, dtype=torch.float32)   

            # if torch.cuda.is_available():
            #     if args.gpu is not None:
            #         labels = labels.cuda(args.gpu, non_blocking=True)
            #         labels = labels.to(dtype=torch.long)
            #     else:
            #         labels = labels.to(device, dtype=torch.long)
            
            outputs = model(images, True)
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = resize(outputs, size=labels.shape[-2:])
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            # print ("preds shape: ", preds.shape)
            # print ("targets shape: ", targets.shape)

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids and args.rank in [-1, 0]:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if args.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1
        score, hist = metrics.get_results()
        # print(metrics.to_str(score))
        # if True:#args.distributed:
        #     hist = torch.tensor(hist).cuda(args.gpu)
        #     torch.distributed.barrier()
        #     torch.distributed.all_reduce(hist)
        #     # hist = reduce_value(hist)
        #     # torch.cuda.synchronize()
        #     hist = hist.cpu().numpy()
        #     acc = np.diag(hist).sum() / hist.sum()
        #     acc_cls = np.diag(hist) / hist.sum(axis=1)
        #     acc_cls = np.nanmean(acc_cls)
        #     iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        #     mean_iu = np.nanmean(iu)
        #     freq = hist.sum(axis=1) / hist.sum()
        #     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        #     cls_iu = dict(zip(range(metrics.n_classes), iu)) 
        #     score = {
        #         "Overall Acc": acc,
        #         "Mean Acc": acc_cls,
        #         "FreqW Acc": fwavacc,
        #         "Mean IoU": mean_iu,
        #         "Class IoU": cls_iu,
        #     }
    return score, ret_samples

    
def get_psum_loss(model, loss, loss_weight=0.1):
    children = list(model.named_children())
    for name, child in children:
        if isinstance(child, QuanConv):
            if child.psum_quan:
                print("name: ", name)
                print("psum loss: ", child.psum_loss)
                loss += (child.psum_loss * loss_weight)
        else:
            get_psum_loss(child, loss)

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    # print("world_size: ", world_size)
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    main()
