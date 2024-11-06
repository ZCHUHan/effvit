from asyncio import base_tasks
from json import load
from math import e
from multiprocessing import reduction
from tomlkit import date
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from models.seg_model_zoo import create_seg_model
from models.utils import resize
from models.nn.quant_lsq import QuanConv, PActFn, PACT, SymmetricQuantFunction
# from utils.loss import ChannelWiseDivergence as CWD_loss
from utils.loss import KLLossSoft as KL_loss
from utils.loss import CriterionCWD as CWD_loss
from utils.loss import MultiClassDiceLoss as Dice_Loss
from utils.scheduler import CosineAnnealingWarmupRestarts as CosineLR

from datetime import datetime, timedelta

from torch.utils.tensorboard import SummaryWriter 
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
    parser.add_argument("--worker", type=int, default=2)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt_teacher", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt_saved", default='default', type=str,
                        help="save checkpoint to")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'kd_pixel', 'kd_channel'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
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
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs,_, _ = model(images, True)
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = resize(outputs, size=labels.shape[-2:])
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            # print ("preds shape: ", preds.shape)
            # print ("targets shape: ", targets.shape)

            metrics.update(targets, preds)
            break 

        score = metrics.get_results()
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

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print("Device: %s" % device)
    print("CUDA_VISIBLE_DEVICES: %s" % os.environ['CUDA_VISIBLE_DEVICES'])


    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.worker,
        drop_last=True, pin_memory=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=opts.worker,
        pin_memory=True)
    
    model_teacher = create_seg_model(opts.model, opts.dataset, weight_url=opts.ckpt_teacher)
    # utils.set_bn_momentum(model_teacher.backbone, momentum=0.01)
    for p in model_teacher.parameters():
        p.requires_grad = False
    # student model
    model = create_seg_model(opts.quan_model, opts.dataset, weight_url=opts.ckpt)


    if device == 'cuda':
        cudnn.benchmark = True
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

       
    if opts.ckpt_teacher is not None and os.path.isfile(opts.ckpt_teacher):
        checkpoint_teacher = torch.load(opts.ckpt_teacher, map_location=torch.device('cpu'))
        if "state_dict" in checkpoint_teacher:
            checkpoint_teacher = checkpoint_teacher["state_dict"]
            print("Teacher Model restored from %s" % opts.ckpt_teacher)
        else:
            checkpoint_teacher = checkpoint_teacher["model_state"]
        try:
            model_teacher.load_state_dict(checkpoint_teacher)    
        except RuntimeError as e:
            print(e)
        model_teacher = nn.DataParallel(model_teacher)
        model_teacher.to(device)
    model_teacher.eval() 

    # print(model)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))
    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization

    def set_log_tensor(m, set_log, name=None, p=False):
        children = list(m.named_children())
        for child_name, child in children:
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, QuanConv):
                if p:
                    print(full_name, child.log_tensor, child.scale_a, child.scale[0], child.scale[2])
                else:
                    child.set_log_tensor(set_log)
            else:
                set_log_tensor(child, set_log, full_name, p)
    torch.set_printoptions(precision=5, sci_mode=False)
    set_log_tensor(model, True)
    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        set_log_tensor(model, True, p=True)
        return

    
if __name__ == '__main__':
    main()
