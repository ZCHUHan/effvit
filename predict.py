from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from models.seg_model_zoo import create_seg_model
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import cv2  # 新增的导入，用于处理视频

def get_argparser():
    parser = argparse.ArgumentParser()

    # 数据集选项
    parser.add_argument("--input", type=str, required=True,
                        help="单张图像、图像目录或视频文件的路径")

    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='训练集名称')

    parser.add_argument("--num_classes", type=int, default=19,
                        help="类别数量（默认：19）")

    parser.add_argument("--model", type=str)
    
    # 训练选项
    parser.add_argument("--save_val_results_to", default=None,
                        help="将分割结果保存到指定目录")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='裁剪验证（默认：False）')

    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='验证的批量大小（默认：1）')

    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--crop_size_h", type=int, default=960)
    
    parser.add_argument("--crop_size_w", type=int, default=1920)

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")

    parser.add_argument("--ckpt", default=None, type=str,
                        help="从检查点恢复")
    return parser
def process_video(video_path, model, device, opts):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化视频写入器
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
        output_path = os.path.join(opts.save_val_results_to, 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 定义转换和解码函数
    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    if opts.dataset.lower() == 'voc':
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        decode_fn = Cityscapes.decode_target

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=frame_count)
    with torch.no_grad():
        model.eval()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 将帧转换为 PIL 图像
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 应用转换
            img_transformed = transform(img).unsqueeze(0).to(device)

            # 模型推理
            pred = model(img_transformed)
            pred = pred.max(1)[1].cpu().numpy()[0]  # HW

            # 解码预测结果
            colorized_preds = decode_fn(pred).astype('uint8')

            # 将预测结果调整为与原始帧相同的尺寸
            colorized_preds = cv2.resize(colorized_preds, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 将颜色空间从 RGB 转换为 BGR（因为 OpenCV 使用 BGR）
            colorized_preds = cv2.cvtColor(colorized_preds, cv2.COLOR_RGB2BGR)

            # 叠加分割结果到原始帧
            alpha = 0.5
            overlay = cv2.addWeighted(frame, 1 - alpha, colorized_preds, alpha, 0)

            # 写入结果帧到输出视频
            if opts.save_val_results_to is not None:
                out.write(overlay)

            pbar.update(1)
    cap.release()
    if opts.save_val_results_to is not None:
        out.release()
    pbar.close()


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    gpu_ids = [int(id) for id in opts.gpu_id.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and len(gpu_ids) > 0 else "cpu")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # print("CUDA_VISIBLE_DEVICES: %s" % os.environ['CUDA_VISIBLE_DEVICES'])
    # Setup random seed
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.set_printoptions(precision=2, sci_mode=True)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        ext = os.path.splitext(opts.input)[-1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 如果输入是视频文件，则调用 process_video 函数
            model = create_seg_model(opts.model, opts.dataset, weight_url=opts.ckpt)
            if opts.ckpt is not None and os.path.isfile(opts.ckpt):
                checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
                try:
                    if "state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["state_dict"])
                    else:
                        model.load_state_dict(checkpoint["model_state"])
                except RuntimeError as e:
                    print(e)
                model.to(device)
                model.eval()
                model = nn.DataParallel(model, device_ids=gpu_ids)
                model.to(device)
                print("Loaded from %s" % opts.ckpt)
                del checkpoint  # free memory
            else:
                print("[!] Retrain Student Model from scratch")
                model = nn.DataParallel(model, device_ids=gpu_ids)
                model.to(device)
            process_video(opts.input, model, device, opts)
            return  # 处理完视频后退出程序
        else:
            image_files.append(opts.input)
    else:
        print("输入路径不是有效的文件或目录")
        return

    if len(image_files) == 0:
        print("在指定的输入路径中未找到图像。")
        return

    model = create_seg_model(opts.model, opts.dataset, weight_url=opts.ckpt)
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        try:
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as e:
            print(e)
        model.to(device)
        model.eval()
        model = nn.DataParallel(model, device_ids=gpu_ids)
        model.to(device)
        print("Loaded from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain Student Model from scratch")
        model = nn.DataParallel(model, device_ids=gpu_ids)
        model.to(device)

    if opts.crop_val:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+'.png'))

if __name__ == '__main__':
    main()
