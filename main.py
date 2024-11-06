from asyncio import base_tasks
from json import load
from math import e
from tomlkit import date
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, ADE20KSegmentation
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
from utils.loss import ChannelWiseDivergence as CWD_loss

from utils.scheduler import CosineAnnealingWarmupRestarts as CosineLR
from datetime import datetime, timedelta

from torch.utils.tensorboard import SummaryWriter 
def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'ade'], help='Name of dataset')
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
    parser.add_argument("--model_s", type=str)
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=300e3,
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

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'kd'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
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
    
    if opts.dataset == 'ade':
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

        train_dst = ADE20KSegmentation(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = ADE20KSegmentation(root=opts.data_root,
                             split='val', transform=val_transform)
    
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs, _, _ = model(images, True)
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = resize(outputs, size=labels.shape[-2:])
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            # print ("preds shape: ", preds.shape)
            # print ("targets shape: ", targets.shape)

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
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

        score = metrics.get_results()
    return score, ret_samples

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'ade':
        opts.num_classes = 42

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    print("CUDA_VISIBLE_DEVICES: %s" % os.environ['CUDA_VISIBLE_DEVICES'])

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    # if opts.dataset == 'voc' and not opts.crop_val:
        # opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.worker,
        drop_last=True, pin_memory=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=opts.worker,
        pin_memory=True)

    summary_writer = SummaryWriter(opts.log_dir)

    # Set up model (all models are 'constructed at network.modeling)
    # model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # if opts.separable_conv and 'plus' in opts.model:
        # network.convert_to_separable_conv(model.classifier)

    # teacher model
    
    model_teacher = create_seg_model(opts.model, opts.dataset, weight_url=opts.ckpt_teacher)
    # utils.set_bn_momentum(model_teacher.backbone, momentum=0.01)
    for p in model_teacher.parameters():
        p.requires_grad = False
    # student model
    model = create_seg_model(opts.model_s, opts.dataset, weight_url=opts.ckpt)


    if device == 'cuda':
        cudnn.benchmark = True
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
    #     {'params': model.classifier.parameters(), 'lr': opts.lr},
    # ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=opts.lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=1)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy == 'cosine':

        scheduler = CosineLR(optimizer, first_cycle_steps=100000, max_lr=opts.lr, min_lr=6e-7, warmup_steps=2000)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.total_itrs)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'kd':
        # KD_loss = CWD_loss(tau=1.0, loss_weight=1.0, size_average=True, ignore_index=255) 
        KD_loss = CWD_loss(tau=4, loss_weight=3, size_average=True, ignore_index=255) 
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        try:
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as e:
            print(e)
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            print("[!] Continue training")
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain Student Model from scratch")
        model = nn.DataParallel(model)
        model.to(device)

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

    print(model)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))
    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    interval_loss_seg = 0
    interval_loss_kd = 0
    avg_iter_duration = timedelta(0)
    # val_score_t, ret_samples_t = validate(
    #         opts=opts, model=model_teacher,
    #         loader=val_loader, device=device,
    #         metrics=metrics,ret_samples_ids=vis_sample_id)
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1
            iter_start_time = datetime.now()
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # print("images",images.shape)
            optimizer.zero_grad()
            # open batchnorm folding
            outputs, outputs_conv, _ = model(images, False)
            # outputs_teacher, outputs_teacher_conv = model_teacher(images)

            if outputs.shape[2:] != labels.shape[-2:]:
                outputs = resize(outputs, size=labels.shape[-2:])
                # print(outputs_teacher)
                # outputs_teacher = resize(outputs_teacher, size=labels.shape[-2:])

            # print("output",outputs.shape)
            # print("labels",labels.shape)
            loss_seg = criterion(outputs, labels)
            # loss_kd = KD_loss(outputs, outputs_teacher, labels)
            # loss_kd = KD_loss(outputs_conv, outputs_teacher_conv)
            loss = loss_seg # + loss_kd
            loss.backward()
            optimizer.step()
            
            
            iter_end_time = datetime.now()
            iter_duration = iter_end_time - iter_start_time
            avg_iter_duration = (avg_iter_duration * (cur_itrs - 1) + iter_duration) / cur_itrs


            remaining_iters = opts.total_itrs - cur_itrs
            estimated_remaining_time = avg_iter_duration * remaining_iters


            estimated_remaining_seconds = estimated_remaining_time.total_seconds()
            estimated_remaining_hours = estimated_remaining_seconds / 3600

            
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            np_loss_seg = loss_seg.detach().cpu().numpy()
            interval_loss_seg += np_loss_seg

            # np_loss_kd = loss_kd.detach().cpu().numpy()
            # interval_loss_kd += np_loss_kd
            
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 50 == 0:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                current_lr = scheduler.get_lr()[0]
                interval_loss = interval_loss / 50
                interval_loss_seg = interval_loss_seg / 50
                # interval_loss_kd = interval_loss_kd / 50

                summary_writer.add_scalar('Loss/train', interval_loss, cur_itrs)
                summary_writer.add_scalar('Loss_seg/train', interval_loss_seg, cur_itrs)
                # summary_writer.add_scalar('Loss_kd/train', interval_loss_kd, cur_itrs)
                summary_writer.add_scalar('LR', current_lr, cur_itrs)

                print(f"[{current_time}] "
                      f"Epoch {cur_epochs}, Itrs {cur_itrs}/{int(opts.total_itrs)}, "
                      f"Loss={interval_loss:.4f}, Loss_seg={interval_loss_seg:.4f}, "
                      f"Loss_kd={interval_loss_kd:.4f}, LR={current_lr:.5e}, "
                      f"Estimated Remaining Time: {estimated_remaining_hours:.2f}h")
                # print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    #   (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0
                interval_loss_seg = 0.0
                # interval_loss_kd = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)

                # print("teacher model scores:\n", metrics.to_str(val_score_t))
                print("student model scores:\n", metrics.to_str(val_score))

                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                summary_writer.add_scalar('mIoU_student/val', val_score['Mean IoU'], cur_itrs)
                summary_writer.add_scalar('Acc_student/val', val_score['Overall Acc'], cur_itrs)

                # summary_writer.add_scalar('mIoU_teacher/val', val_score_t['Mean IoU'], cur_itrs)
                # summary_writer.add_scalar('Acc_teacher/val',  val_score_t['Overall Acc'], cur_itrs)

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()

