import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from inspect import signature
from typing import Any, List, Optional, Tuple, Union, Dict, Callable, Type
from torch.autograd import Function
import decimal

__all__ = [
    "ConvLayer",
    "UpSampleLayer",
    "IdentityLayer",
    "DSConv",
    "MBConv",
    "LiteMSA",
    "EfficientViTBlock",
    "ResidualBlock",
    "DAGBlock",
    "OpSequential",
    "QuantAct",
    "build_act",
    "build_norm",
    "get_same_padding",
    "merge_tensor",
    "resize",
    "build_kwargs_from_config",
    "load_state_dict_from_file",
    "list_sum",
    "val2list",
    "val2tuple",
]

#################################################################################
#                             Basic Layers                                      #
#################################################################################

    
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        per_channel,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout_rate=0,
        norm="bn2d",
        act_func="relu",
        weight_bit=8,
        activation_bit=8,
        eval_mode=False
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation
        self.eval_mode=eval_mode
        self.dropout = nn.Dropout2d(dropout_rate, inplace=False) if dropout_rate > 0 else None
        # self.norm = build_norm(norm, num_features=out_channels)
        self.norm = norm
        self.activation_bit = activation_bit
        if norm:
            self.conv = QConvBNReLU(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                per_channel=per_channel,
                stride=(stride, stride),
                padding=padding,
                dilation=(dilation, dilation),
                groups=groups,
                bias=use_bias,
                weight_bit=weight_bit, # TODO
                activation_bit=activation_bit,
                eval_mode=eval_mode
            )
        else:
            self.conv = QuanConv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                per_channel=per_channel,
                stride=(stride, stride),
                padding=padding,
                dilation=(dilation, dilation),
                groups=groups,
                bias=use_bias,
                weight_bit=weight_bit, # TODO
                activation_bit=activation_bit,
                eval_mode=eval_mode
            )
        self.act_func = act_func
        if self.act_func == 'relu' or self.act_func == 'relu6':
            print('bulid pact')
            self.act =  LearnedClippedLinearQuantization(activation_bit, 6.0, eval_mode=eval_mode, dequantize=False, inplace=False)
            # self.act = nn.ReLU6()
        elif self.act_func == 'hswish':
            self.act = build_act(act_func)
        #     self.act = torch.nn.quantized.functional.hardswish
        self.quan_act_in = QuantAct(activation_bit=activation_bit)
        self.quan_act_out = QuantAct(activation_bit=activation_bit)
        if self.act_func:
            self.quan_actfn_out = QuantAct(activation_bit=activation_bit)
    
    def forward(self, x: torch.Tensor, quantized_bit,x_scale=None, iter=0) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x, x_scale, x_bit = self.quan_act_in(x, x_scale=x_scale, quant_mode=True, quantized_bit=quantized_bit) # 16 bit, one 32 bit?
        x, x_scale, x_bit = self.conv(x, x_scale=x_scale, quant_mode=True) # if fuse, 32 bit; else, 24 bit; now per_channel
        if not self.training and self.eval_mode and x_scale is not None and torch.max(torch.abs(x)) > 2**(x_bit-1):
            print(x)
            print('conv conv error')
            exit()
        x, x_scale, x_bit = self.quan_act_out(x, x_scale=x_scale, quant_mode=True, quantized_bit=x_bit) # return to 16 bit
        if not self.training and self.eval_mode and x_scale is not None and torch.max(torch.abs(x)) > 2**(x_bit-1):
            print(x, torch.max(torch.abs(x)))
            print('conv act_out error')
            exit()
        if self.act_func == 'relu' or self.act_func == 'relu6': # actually do not used
            x, x_scale = self.act(x, x_scale)
        elif self.act_func is not None:
            if x_scale is not None:
                if self.training or (not self.training and not self.eval_mode): # 似乎训练的时候不需要对这个进行quant，因为这里没有需要训练的目标？
                    x = x / x_scale
                    x_clamp = torch.clamp(x + torch.round(3 / x_scale), torch.tensor(0.0, device='cuda'), torch.round(6 / x_scale)) / 6
                    x_scale = x_scale * x_scale
                    x = x * x_clamp * x_scale # 32 bit
                    x_bit = 32
                    x, x_scale, x_bit = self.quan_actfn_out(x, x_scale=x_scale, quant_mode=True, quantized_bit=x_bit) # return to 16 bit
                else:
                    x_clamp = torch.clamp(x + torch.round(3 / x_scale), torch.tensor(0.0, device='cuda'), torch.round(6 / x_scale)) / 6
                    x = x * x_clamp
                    x_scale = x_scale * x_scale
                    x_bit = 32
                    x, x_scale, x_bit = self.quan_actfn_out(x, x_scale=x_scale, quant_mode=True, quantized_bit=x_bit) # return to 16 bit
                    if not self.training and self.eval_mode and x_scale is not None and torch.max(torch.abs(x)) > 2**(x_bit-1):
                        print(x)
                        print('conv actfn_out error')
                        exit()
            else:
                x = F.hardswish(x)
        x_out_bit = self.activation_bit
        if not self.training and self.eval_mode and x_scale is not None and torch.max(torch.abs(x)) > 2**(x_out_bit-1):
            print(x, torch.max(torch.abs(x)))
            print('conv out error')
            exit()
        return x, x_scale, x_out_bit # 16 bit

    
class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: Union[int, Tuple[int, int], List[int], None] = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, iter=0) -> torch.Tensor:
        return resize(x, self.size, self.factor, self.mode, self.align_corners)



class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor,quantized_bit,x_scale=None,iter=0) -> torch.Tensor:
        if x.grad is not None:
            print(x.grad.flatten()[0].item())
        return x, x_scale, quantized_bit


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        per_channel,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
        weight_bit=8,
        activation_bit=8,
        eval_mode=False
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.eval_mode=eval_mode
        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            per_channel,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            per_channel,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )
    
    def forward(self, x: torch.Tensor, quantized_bit,x_scale=None, iter=0) -> torch.Tensor:
        x, x_scale, x_out_bit = self.depth_conv(x, x_scale=x_scale, quantized_bit=quantized_bit)
        if not self.training and self.eval_mode and x_scale is not None and torch.max(torch.abs(x)) > 2**(x_out_bit-1):
            print(x)
            print('ds depth error')
            exit()
        x, x_scale, x_out_bit = self.point_conv(x, x_scale=x_scale, quantized_bit=x_out_bit)
        if not self.training and self.eval_mode and x_scale is not None and torch.max(torch.abs(x)) > 2**(x_out_bit-1):
            print(x)
            print('ds point error')
            exit()
        return x, x_scale, x_out_bit


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int, # 32
        out_channels: int, # 32 
        per_channel,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6, # 4
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
        weight_bit=8,
        activation_bit=8,
        eval_mode=False
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            per_channel,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            per_channel,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            per_channel,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )

    def forward(self, x: torch.Tensor,quantized_bit,x_scale=None,iter=0) -> torch.Tensor:
        x, x_scale, x_out_bit = self.inverted_conv(x, x_scale=x_scale, quantized_bit=quantized_bit)
        x, x_scale, x_out_bit = self.depth_conv(x, x_scale=x_scale, quantized_bit=x_out_bit)
        x, x_scale, x_out_bit = self.point_conv(x, x_scale=x_scale, quantized_bit=x_out_bit)
        return x, x_scale, x_out_bit



def SoftThreshold(x, th, status = True): # status = True means it is training now, False means evaluation
    c = 1000
    if status == True:
        s = 1000
        # x is the attention score tensor
        # alpha is a learnable parameter
        condition = x >= th
        x = torch.where(condition, x * torch.tanh(s*(x-th)), c * torch.tanh(s*(x-th)))
    else:
        # condition = x >= th
        # x = torch.where(condition, x, torch.zeros_like(x))
        result = torch.lt(x, th).sum() / x.numel()
        y = torch.zeros_like(x) # 一个和x形状相同，值为-1000的tensor
        x = torch.where(x > th, x, y) # output是一个根据条件选择x或y中元素的tensor
    return x


def count_prune_ratio(x, th):
    # 获取占比
    ratio = torch.lt(x, th).sum() / x.numel()
    return ratio

def l0_regularizer_loss(x, th):
    B, _, _, _ = x.shape
    k = 100
    c = 1000
    alpha = 1
    x_s = torch.sigmoid(k*(x + c - alpha))
    total = torch.numel(x)
    unprune_ratio = torch.sum(x_s) / (total) # unprune ratio, use ratio for loss ? TODO
    
    prune_ratio_max = 0.80
    unprune_ratio_min = 1 - prune_ratio_max
    condition = unprune_ratio >= unprune_ratio_min
    l0_loss = torch.where(condition, unprune_ratio, 10*(unprune_ratio_min - unprune_ratio)) # un_ratio >= 0.3, loss = un_ratio; un_ratio < 0.3, loss = 0.3 - un_ratio
    num_nonzeros = torch.count_nonzero(torch.lt(x, th))
    
    # 获取占比
    # if not torch.sum(x_s) < total:
    #     print(x)
    # print(1-num_nonzeros / total, " ", unprune_ratio, " ", torch.sum(x_s), " ", total)
    return unprune_ratio # if need max prune ratio, should return "l0_loss", otherwise unprune_ratio

def eliminate_col_traditional(k, ratio, th=0.0):
    B, num_heads, N, _, C = k.shape
    count_all = 0
    count = 0
    for q in range(B):
        for i in range(num_heads):
            for j in range(N):
                count_all += 1
                # 获取当前维度上的切片
                k_slice = k[q, i, j, :, :]
                # 统计该切片中0的个数
                num_zeros = torch.sum(k_slice <= 0)
                # 如果0的个数大于该切片元素总数的50%
                if num_zeros > (k_slice.size()[0] * k_slice.size()[1] * (1-ratio)):
                    # 将该切片置为0
                    k[q, i, j, :, :] = 0
                    count += 1
    return k, count / count_all


def eliminate_col(k, ratio, th=0.0):
    num_non_zeros = torch.count_nonzero(k > th, dim=(3, 4))
    # Compute the total number of elements in each head and position slice
    total_elements = k.shape[3] * k.shape[4]
    mask = (num_non_zeros / total_elements) < ratio # if num_non_zeros ratio less than 0.25, than true
    # 将bool类型的tensor转换为float类型的tensor
    float_tensor = mask.float()
    # 计算True的比例
    mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1, 1).expand_as(k)
    mask_not = torch.logical_not(mask)
    k = k * mask_not
    return k, float_tensor.sum() / float_tensor.numel()

def test_col(k, ratio, th=0.0):
    num_non_zeros = torch.count_nonzero(k > th, dim=(3, 4))
    # Compute the total number of elements in each head and position slice
    total_elements = k.shape[3] * k.shape[4]
    mask = (num_non_zeros / total_elements) < ratio # if num_non_zeros ratio less than 0.25, than true
    # 将bool类型的tensor转换为float类型的tensor
    float_tensor = mask.float()
    return float_tensor.sum() / float_tensor.numel()

def eliminate_row(k, ratio, th=0.0): # input kv: BHCC, H refers to head*multiscale
    num_non_zeros = torch.count_nonzero(k > th, dim=(3))
    # Compute the total number of elements in each head and position slice
    total_elements = k.shape[3]
    mask = (num_non_zeros / total_elements) < ratio # if num_non_zeros ratio less than 0.25, than true
    # 将bool类型的tensor转换为float类型的tensor
    float_tensor = mask.float()
    # 计算True的比例
    mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], 1).expand_as(k)
    mask_not = torch.logical_not(mask)
    k = k * mask_not
    return k, float_tensor.sum() / float_tensor.numel()


class LiteMSA(nn.Module):
    r""" Lightweight multi-scale attention """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        per_channel,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: Tuple[int, ...] = (5,),
        weight_bit=8,
        activation_bit=8,
        eval_mode=False
    ):
        super(LiteMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)
        self.eval_mode=eval_mode
        self.heads = heads
        total_dim = heads * dim
        self.total_dim = total_dim
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.weight_bit = weight_bit
        self.activation_bit = activation_bit
        self.dim = dim # this dim is head_dim
        # self.qkv = ConvLayer(
        #     in_channels, # 64, 64, 128, 128
        #     3 * total_dim, # 64, 64, 128, 128
        #     per_channel,
        #     1,
        #     use_bias=use_bias[0], # use_bias = False, False
        #     norm=norm[0], # None
        #     act_func=act_func[0], # None
        #     weight_bit=weight_bit,
        #     activation_bit=activation_bit,
        #     eval_mode=eval_mode
        # )
        self.q = ConvLayer(
            in_channels, # 64, 64, 128, 128
            total_dim, # 64, 64, 128, 128
            per_channel,
            1,
            use_bias=use_bias[0], # use_bias = False, False
            norm=norm[0], # None
            act_func=act_func[0], # None
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )
        self.k = ConvLayer(
            in_channels, # 64, 64, 128, 128
            total_dim, # 64, 64, 128, 128
            per_channel,
            1,
            use_bias=use_bias[0], # use_bias = False, False
            norm=norm[0], # None
            act_func=act_func[0], # None
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )
        self.v = ConvLayer(
            in_channels, # 64, 64, 128, 128
            total_dim, # 64, 64, 128, 128
            per_channel,
            1,
            use_bias=use_bias[0], # use_bias = False, False
            norm=norm[0], # None
            act_func=act_func[0], # None
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )
        self.aggreg_q = nn.ModuleList(
            [
                nn.Sequential(
                    QuanConv2d(
                        total_dim, total_dim, scale, 
                        per_channel=per_channel,padding=get_same_padding(scale), groups=total_dim, bias=use_bias[0],
                        weight_bit=weight_bit,
                        activation_bit=activation_bit,
                        eval_mode=eval_mode
                    ),
                    QuanConv2d(total_dim, total_dim, 1, 
                    per_channel=per_channel,groups=heads, bias=use_bias[0], 
                    weight_bit=weight_bit,
                    activation_bit=activation_bit,
                    eval_mode=eval_mode), # heads=4,4,8,8
                )
                for scale in scales
            ]
        )
        self.aggreg_k = nn.ModuleList(
            [
                nn.Sequential(
                    QuanConv2d(
                        total_dim, total_dim, scale, 
                        per_channel=per_channel,padding=get_same_padding(scale), groups=total_dim, bias=use_bias[0],
                        weight_bit=weight_bit,
                        activation_bit=activation_bit,
                        eval_mode=eval_mode
                    ),
                    QuanConv2d(total_dim, total_dim, 1, 
                    per_channel=per_channel,groups=heads, bias=use_bias[0], 
                    weight_bit=weight_bit,
                    activation_bit=activation_bit,
                    eval_mode=eval_mode), # heads=4,4,8,8
                )
                for scale in scales
            ]
        )
        self.aggreg_v = nn.ModuleList(
            [
                nn.Sequential(
                    QuanConv2d(
                        total_dim, total_dim, scale, 
                        per_channel=per_channel,padding=get_same_padding(scale), groups=total_dim, bias=use_bias[0],
                        weight_bit=weight_bit,
                        activation_bit=activation_bit,
                        eval_mode=eval_mode
                    ),
                    QuanConv2d(total_dim, total_dim, 1, 
                    per_channel=per_channel,groups=heads, bias=use_bias[0], 
                    weight_bit=weight_bit,
                    activation_bit=activation_bit,
                    eval_mode=eval_mode), # heads=4,4,8,8
                )
                for scale in scales
            ]
        )
        self.quan_act_q1 = QuantAct(activation_bit=activation_bit)
        self.quan_act_q2 = QuantAct(activation_bit=activation_bit)
        self.quan_act_k1 = QuantAct(activation_bit=activation_bit)
        self.quan_act_k2 = QuantAct(activation_bit=activation_bit)
        self.quan_act_v1 = QuantAct(activation_bit=activation_bit)
        self.quan_act_v2 = QuantAct(activation_bit=activation_bit)
        self.quan_cat_q = QuantCat(activation_bit=activation_bit, all_positive=True)
        self.quan_cat_k = QuantCat(activation_bit=activation_bit, all_positive=True)
        self.quan_cat_v = QuantCat(activation_bit=activation_bit)
        # self.aggreg = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             QuanConv2d(
        #                 3 * total_dim, 3 * total_dim, scale, 
        #                 per_channel=per_channel,padding=get_same_padding(scale), groups=3 * total_dim, bias=use_bias[0],
        #                 weight_bit=weight_bit,
        #                 activation_bit=activation_bit,
        #                 eval_mode=eval_mode
        #             ),
        #             QuanConv2d(3 * total_dim, 3 * total_dim, 1, 
        #             per_channel=per_channel,groups=3 * heads, bias=use_bias[0], 
        #             weight_bit=weight_bit,
        #             activation_bit=activation_bit,
        #             eval_mode=eval_mode), # 3*heads=12,12,24,24
        #         )
        #         for scale in scales
        #     ]
        # )
        # self.kernel_func = build_act(kernel_func, inplace=False)
        # self.kernel_func_q = nn.ReLU6(inplace=False)
        # self.kernel_func_k = nn.ReLU6(inplace=False)
        self.kernel_func_q = LearnedClippedLinearQuantization(activation_bit, 6.0, eval_mode=eval_mode, dequantize=False, inplace=False)
        self.kernel_func_k = LearnedClippedLinearQuantization(activation_bit, 6.0, eval_mode=eval_mode, dequantize=False, inplace=False)
        self.k_col_ratio = 0
        self.soft_th_q = nn.Parameter(torch.tensor(0.05))
        self.soft_th_k = nn.Parameter(torch.tensor(0.05))
        self.prune_ratio_q = 0
        self.prune_ratio_k = 0
        self.eliminate_ratio_q = 0
        self.eliminate_ratio_k = 0
        self.l0_regularizer_loss_q = 0
        self.l0_regularizer_loss_k = 0
        self.org_zero_ratio_q = 0
        self.org_zero_ratio_k = 0
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels, # 64, 64, 128, 128
            per_channel,
            1,
            use_bias=use_bias[1], # False
            norm=norm[1], # bn2d
            act_func=act_func[1], # None
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )
        self.quan_act_kv = QuantAct(activation_bit=32)
        self.quan_act_proj = QuantAct(activation_bit=activation_bit) #  TODO
        # print("heads", heads) # b1 first three  8, 1st conv, kernel 5, groups 24; 2nd conv kernel 1, groups 48 384->384
        # b1 second four 8, 1st conv kernel 5, groups 24; 2nd conv kernel 1, groups 48 768->768 
        # print("kernel_func", kernel_func) # relu
    
    def forward(self, x: torch.Tensor, quantized_bit,x_scale=None, iter=0) -> torch.Tensor: # input x 16 bit
        if torch.any(torch.isnan(x)):
            print("x has nan", self.heads)
            exit()
        B, _, H, W = list(x.size())
        # 真正的q中0,1,2是multiscale 0到128共8个heads的1,4,7个，所以q的0-8是1,4,7...
        q, q_scale, q_out_bit = self.q(x, x_scale=x_scale, quantized_bit=quantized_bit)
        k, k_scale, k_out_bit = self.k(x, x_scale=x_scale, quantized_bit=quantized_bit)
        v, v_scale, v_out_bit = self.v(x, x_scale=x_scale, quantized_bit=quantized_bit)
        q, q_scale, q_out_bit = self.quan_act_q1(q, x_scale=q_scale, quantized_bit=q_out_bit)
        k, k_scale, k_out_bit = self.quan_act_k1(k, x_scale=k_scale, quantized_bit=k_out_bit)
        v, v_scale, v_out_bit = self.quan_act_v1(v, x_scale=v_scale, quantized_bit=v_out_bit)
        q_aggreg, q_aggreg_scale, q_aggreg_bit = self.aggreg_q[0][0](q, q_scale)
        k_aggreg, k_aggreg_scale, k_aggreg_bit = self.aggreg_k[0][0](k, k_scale)
        v_aggreg, v_aggreg_scale, v_aggreg_bit = self.aggreg_v[0][0](v, v_scale)
        q_aggreg, q_aggreg_scale, q_aggreg_bit = self.quan_act_q2(q_aggreg, x_scale=q_aggreg_scale, quantized_bit=q_aggreg_bit, quant_mode=True)
        k_aggreg, k_aggreg_scale, k_aggreg_bit = self.quan_act_k2(k_aggreg, x_scale=k_aggreg_scale, quantized_bit=k_aggreg_bit, quant_mode=True)
        v_aggreg, v_aggreg_scale, v_aggreg_bit = self.quan_act_v2(v_aggreg, x_scale=v_aggreg_scale, quantized_bit=v_aggreg_bit, quant_mode=True)
        q_aggreg, q_aggreg_scale, q_aggreg_bit = self.aggreg_q[0][1](q_aggreg, q_aggreg_scale)
        k_aggreg, k_aggreg_scale, k_aggreg_bit = self.aggreg_k[0][1](k_aggreg, k_aggreg_scale)
        v_aggreg, v_aggreg_scale, v_aggreg_bit = self.aggreg_v[0][1](v_aggreg, v_aggreg_scale)
        q, q_scale, q_bit = self.quan_cat_q([q, q_aggreg], [q_scale, q_aggreg_scale], quant_mode=True)
        k, k_scale, k_bit = self.quan_cat_k([k, k_aggreg], [k_scale, k_aggreg_scale], quant_mode=True)
        v, v_scale, v_bit = self.quan_cat_v([v, v_aggreg], [v_scale, v_aggreg_scale], quant_mode=True)
        q = torch.reshape(q, (1,-1,self.dim,H*W,),)
        q = torch.transpose(q, -1, -2)
        k = torch.reshape(k, (1,-1,self.dim,H*W,),)
        k = torch.transpose(k, -1, -2)
        v = torch.reshape(v, (1,-1,self.dim,H*W,),)
        v = torch.transpose(v, -1, -2)
        # lightweight global attention
        q, q_scale, q_bit = self.kernel_func_q(q, q_scale, quant_mode=True) # quant_act may add after this one?
        k, k_scale, k_bit = self.kernel_func_k(k, k_scale, quant_mode=True) # all positive 
        # # 计算为0的比例
        # k = torch.unsqueeze(k, dim=-2)
        # v = torch.unsqueeze(v, dim=-2)
        
        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 1), mode="constant", value=1)
        if k_scale is not None:
            kv_scale = k_scale * v_scale
            kv = torch.matmul(trans_k, v)
        else:
            kv, kv_scale = torch.matmul(trans_k, v), None
        kv_bit = 32
        # kv = torch.sum(kv, dim=2, keepdim=True).reshape(
        #     B, -1, self.dim, (self.dim+1)).contiguous() # bit: TODO
        kv_quan, kv_scale, kv_bit = self.quan_act_kv(kv, x_scale=kv_scale, quant_mode=True, quantized_bit=kv_bit) # return to 16 bit, kv too large
        if kv_scale is not None:
            out_scale = q_scale * kv_scale
            out = torch.matmul(q, kv_quan)
        else:
            out, out_scale = torch.matmul(q, kv_quan), None
        out_bit = 32
        # print(out_scale, scale_tmp) problem here
        out = out[..., :-1] / (out[..., -1:] + 1e-15) # remove the last channel, 17 -> 16
        out_quan, out_scale, out_bit = self.quan_act_proj(out, x_scale=out_scale, quant_mode=True, quantized_bit=out_bit) # return to 16 bit problem here
        # final projecttion
        out_quan = torch.transpose(out_quan, -1, -2)
        out_quan = torch.reshape(out_quan, (B, -1, H, W))
        proj, proj_scale, proj_bit = self.proj(out_quan, x_scale=out_scale, quantized_bit = out_bit) # returned to 16 bit in conv layer 可能有问题，因为quan5用了32bit
        return proj, proj_scale, proj_bit
    
class EfficientViTBlock(nn.Module):
    def __init__(self, in_channels: int, per_channel,heads_ratio: float = 1.0, dim=32, expand_ratio: float = 4, norm="bn2d", act_func="hswish", weight_bit=8, activation_bit=8,eval_mode=False):
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMSA(
                in_channels=in_channels,
                out_channels=in_channels,
                per_channel=per_channel,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                weight_bit=weight_bit,
                activation_bit=activation_bit,
                eval_mode=eval_mode
            ),
            IdentityLayer(),
            eval_mode=eval_mode,
            activation_bit=activation_bit,
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            per_channel=per_channel,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
            weight_bit=weight_bit,
            activation_bit=activation_bit,
            eval_mode=eval_mode
        )
        self.local_module = ResidualBlock(
            local_module, 
            IdentityLayer(), 
            eval_mode=eval_mode,
            activation_bit=activation_bit,
        )
    
    def forward(self, x: torch.Tensor, quantized_bit,x_scale=None, iter=0) -> torch.Tensor:
        x, x_scale, x_bit = self.context_module(x, x_scale=x_scale,iter=iter,quantized_bit=quantized_bit)
        x, x_scale, x_bit = self.local_module(x, x_scale=x_scale,iter=iter, quantized_bit=x_bit)
        return x, x_scale, x_bit


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    test_res_num_count = 0
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
        activation_bit=16,
        eval_mode=False
    ):
        super(ResidualBlock, self).__init__()
        self.eval_mode=eval_mode
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)
        self.res_quant_act = QuantAct(activation_bit=activation_bit)
        if self.shortcut is not None: 
            ResidualBlock.test_res_num_count += 1
            self.test_res_num = ResidualBlock.test_res_num_count
        else:
            self.test_res_num = 0
        

    def forward_main(self, x: torch.Tensor, quantized_bit,x_scale=None,iter=0) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x, x_scale=x_scale,iter=iter,quantized_bit=quantized_bit)
        else:
            return self.main(self.pre_norm(x), iter=iter,quantized_bit=quantized_bit)

    def forward(self, x: torch.Tensor, x_scale,quantized_bit,iter=0) -> torch.Tensor:
        if torch.all(x == 0):
            print('res input 0')
        elif torch.any(torch.isnan(x)): 
            print('res input nan')
        if self.main is None:
            res = x
            res_scale = x_scale
            res_bit = quantized_bit
        elif self.shortcut is None:
            res, res_scale, res_bit = self.forward_main(x, x_scale=x_scale,iter=iter,quantized_bit=quantized_bit)
        else:
            res_forward_main, res_forward_main_scale, res_bit = self.forward_main(x, x_scale=x_scale,iter=iter,quantized_bit=quantized_bit)
            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            res_shortcut, res_shortcut_scale, res_bit = self.shortcut(x,x_scale=x_scale,quantized_bit=quantized_bit) # TODO
            q = True if (self.test_res_num >= 1 and self.test_res_num <= 1)else False
            # if q: print(self.forward_main)
            res, res_scale, res_bit = self.res_quant_act(res_forward_main, x_scale=res_forward_main_scale, quant_mode=q , quantized_bit=res_bit, identity=res_shortcut, identity_scaling_factor=res_shortcut_scale) # TODO
            # print("2:{}".format(torch.cuda.memory_allocated(0)))
            if self.post_act: # unused
                print('use post_act')
                res, res_scale, res_bit = self.post_act(res,res_scale, quantized_bit=res_bit)
        
        if torch.all(res == 0):
            print('res output 0')
        elif torch.any(torch.isnan(res)):
            print('res output nan') 
            # raise ValueError('res output nan')
        if self.eval_mode and not self.training and res_scale is not None and torch.max(torch.abs(res)) > 2**(res_bit-1):
            print(res)
            print('res out error')
            exit()
        return res, res_scale, res_bit


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: Dict[str, nn.Module],
        merge_mode: str,
        post_input: Optional[nn.Module],
        middle: nn.Module,
        outputs: Dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()
        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge_mode = merge_mode
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        feat = merge_tensor(feat, self.merge_mode, dim=1)
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)
    
    def forward(self, x: torch.Tensor, quantized_bit, x_scale=None,iter=0) -> torch.Tensor:
        x_bit = quantized_bit
        for op in self.op_list:
            x, x_scale, x_bit = op(x, x_scale=x_scale,iter=iter, quantized_bit=x_bit)
        return x, x_scale, x_bit
    


class QuanConv2d(nn.Module):
    def __init__(self, 
                in_channels,
                out_channels,
                kernel_size,
                per_channel,
                stride= 1,
                padding= 0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
                weight_bit=8, 
                activation_bit=8,
                eval_mode=False
        ):
        super().__init__()
        self.activation_bit = activation_bit
        self.weight_bit= weight_bit
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.zeros([out_channels, in_channels // groups, kernel_size[0], kernel_size[1]]))
        self.weight_bit = weight_bit
        self.quan_w_fn = LsqQuanParam(bit=weight_bit, per_channel=per_channel, all_positive=False, symmetric=True, per_channel_num=out_channels)
        self.weight_function = SymmetricQuantFunction.apply
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            # self.bias = None
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.init_state = 0
        self.eval_mode = eval_mode

    def forward(self, x, x_scale=None, quant_mode = False):
        if not quant_mode:
            return F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups), None, 32
        if not self.training and self.eval_mode: # fix point
            weight_q, weight_scale = self.quan_w_fn(self.weight)
            conv_scale = x_scale * weight_scale # 要搞成m, e形式的
            bias_q = self.weight_function(self.bias, 32, False, conv_scale)
            return F.conv2d(x, weight=weight_q, bias=bias_q, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups), conv_scale, 32
        # assert x_scale is not None and x_scale.shape == (1,), (
        #     "Input activation to the QuanConv2d layer should be globally (non-channel-wise) quantized. "
        #     "Please add a QuantAct layer with `per_channel = True` before this QuantAct layer"
        # )
        activation, activation_scale = x, x_scale
        if self.init_state == 0: # TODO: what if resume or load is perform?
            if self.quan_w_fn.init_from(self.weight) == True: print('weight init from the first input')
            self.init_state = 1
        weight_q, weight_scale = self.quan_w_fn(self.weight) # weight must be quantized at the current layer
        # weight_q = self.weight_function(
        #     self.weight, self.weight_bit, None, weight_scale
        # )
        conv_scale = (weight_scale * activation_scale) if activation_scale is not None else weight_scale
        if self.bias is not None:
            # bias_q = quantize_according_to_parameters(self.bias, conv_scale, 32)
            bias_q = self.weight_function(self.bias, 32, False, conv_scale)
            # bias_q = None
        else:
            bias_q = None
        activation_q = (activation / activation_scale) if activation_scale is not None else activation
        conv_scale = conv_scale.view(1, -1, 1, 1)
        return F.conv2d(activation_q, weight=weight_q, bias=bias_q, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) * conv_scale, conv_scale, 32

def has_close_enough_values(x, y, threshold=0.01, percentage=0.8):
    rel_err = torch.abs(x - y) / torch.abs(y)
    close_count = torch.sum(rel_err < threshold).item()
    total_count = x.numel()
    return close_count / total_count > percentage

class QConvBNReLU(nn.Module):
    def __init__(self, 
                in_channels,
                out_channels,
                kernel_size,
                per_channel,
                stride= 1,
                padding= 0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                device=None,
                dtype=None,
                weight_bit=8, 
                activation_bit=8,
                eval_mode=False
        ):
        super().__init__()
        self.activation_bit = activation_bit
        self.weight_bit = weight_bit
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.zeros([out_channels, in_channels // groups, kernel_size[0], kernel_size[1]]))
        self.weight_bit = weight_bit
        self.quan_w_fn = LsqQuanParam(bit=weight_bit, per_channel=per_channel, all_positive=False, symmetric=True, per_channel_num=out_channels)
        self.weight_function = SymmetricQuantFunction.apply
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.bn_track_running_stats = True
        if self.bn_track_running_stats:
            self.register_buffer('bn_running_mean', torch.zeros(out_channels))
            self.register_buffer('bn_running_var', torch.ones(out_channels))
            self.register_buffer('bn_num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('bn_num_batches_tracked', None)
        self.bn_momentum = 0.1 # torch.nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.bn_affine = True
        self.bn_eps = 1e-5 # nn.Parameter(torch.tensor(1e-5), requires_grad=False)
        self.init_state = 0
        self.eval_mode = eval_mode
    
    def fold_bn(self, mean, std):
        if self.bn_affine:
            gamma_ = self.bn_weight / std  # 这一步计算gamma' 
            weight = self.weight * gamma_.view(self.out_channels, 1, 1, 1)
            if self.bias is not None:
                bias = gamma_ * self.bias - gamma_ * mean + self.bn_bias
            else:
                bias = self.bn_bias - gamma_ * mean
        else:  # affine为False的情况，gamma=1, beta=0
            gamma_ = 1 / std
            weight = self.weight * gamma_
            if self.bias is not None:
                bias = gamma_ * self.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
            
        return weight, bias
        
    def forward(self, x, x_scale=None, quant_mode = False):
        # print(self.weight)
        # print('bn',x.shape, x_scale.shape)
        if not quant_mode:
            # print(self.weight, self.bias) # 和pth的weight是一样的
            weight, bias = self.fold_bn(self.bn_running_mean, torch.sqrt(self.bn_running_var + self.bn_eps))
            return F.conv2d(x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups), None, 32 # bit TODO
        if not self.training and self.eval_mode: # fix point
            weight, bias = self.fold_bn(self.bn_running_mean, torch.sqrt(self.bn_running_var + self.bn_eps))
            weight_q, weight_scale = self.quan_w_fn(weight)
            conv_scale = x_scale * weight_scale # 要搞成m, e形式的
            bias_q = self.weight_function(self.bias, 32, False, conv_scale)
            return F.conv2d(x, weight=weight_q, bias=bias_q, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups), conv_scale, 32
        # assert x_scale is not None and x_scale.shape == (1,), (
        #     "Input activation to the QuanConv2dBN layer should be globally (non-channel-wise) quantized. "
        #     "Please add a QuantAct layer with `per_channel = True` before this QuantAct layer"
        # )
        if self.training: # 开启BN层训练
            y = F.conv2d(x, self.weight, self.bias, 
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            groups=self.groups)
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.contiguous().view(self.out_channels, -1) # CNHW -> (C,NHW)，这一步是为了方便channel wise计算均值和方差
            mean = y.mean(1)
            var = y.var(1)
            self.bn_running_mean = \
                self.bn_momentum * self.bn_running_mean + \
                (1 - self.bn_momentum) * mean
            self.bn_running_var = \
                self.bn_momentum * self.bn_running_var + \
                (1 - self.bn_momentum) * var
        else: # BN层不更新
            mean = self.bn_running_mean
            var = self.bn_running_var
        std = torch.sqrt(var + self.bn_eps)
        weight, bias = self.fold_bn(mean, std) # no problem
        activation, activation_scale = x, x_scale
        if self.init_state == 0: # TODO: what if resume or load is perform?
            if self.quan_w_fn.init_from(weight) == True: print('weight init from the first input')
            self.init_state = 1
        weight_q, weight_scale = self.quan_w_fn(weight) # weight must be quantized at the current layer
        # weight_q = self.weight_function(
        #     weight, self.weight_bit, None, weight_scale
        # )
        conv_scale = (weight_scale * activation_scale) if activation_scale is not None else weight_scale
        # bias_q = quantize_according_to_parameters(bias, conv_scale, 32)
        bias_q = self.weight_function(bias, 32, None, conv_scale) # 要64才够???
        activation_q = (activation / activation_scale) if activation_scale is not None else activation
        # after F.conv, BCHW, scale should be BCHW (if per_channel, should be 1, 8, 1, 1)
        conv_scale = conv_scale.view(1, -1, 1, 1)
        return F.conv2d(activation_q, weight=weight_q, bias=bias_q, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) * conv_scale, conv_scale, 32
     
class QuantAct(nn.Module):
    """
    Quantizes the given activation.

    Args:
        activation_bit (`int`):
            Bitwidth for the quantized activation.
        act_range_momentum (`float`, *optional*, defaults to `0.95`):
            Momentum for updating the activation quantization range.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether to or not use channel-wise quantization.
        channel_len (`int`, *optional*):
            Specify the channel length when set the *per_channel* True.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(self, activation_bit=8, eval_mode=False):
        super().__init__()
        self.init_state = 0
        self.activation_bit = activation_bit
        self.per_channel = False
        self.percentile = False
        self.quan_a_fn = LsqQuanParam(activation_bit, per_channel=self.per_channel, all_positive=False, symmetric=True) # TODO
        self.eval_mode = eval_mode

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(activation_bit={self.activation_bit}, "
        )

    def forward(
        self,
        x,
        quantized_bit,
        x_scale=None,
        identity=None,
        identity_scaling_factor=None,
        quant_mode = False
    ):
        x_act = x if identity is None else identity + x
        if not quant_mode:
            return x_act, None, self.activation_bit
        elif quantized_bit is not None and quantized_bit == self.activation_bit and identity is None:
            return x_act, x_scale, self.activation_bit
        if not self.training and self.eval_mode:
            print('eval')
            if x_scale is not None: # x_scale in may be channel_wise
                quant_act_int = FixedPointMulInference.apply(
                    x,
                    x_scale,
                    self.activation_bit,
                    self.quan_a_fn.s,
                    identity,
                    identity_scaling_factor,
                ) # TODO
                return quant_act_int, self.quan_a_fn.s, self.activation_bit
        if self.init_state == 0: # TODO: what if resume or load is perform?
            if self.quan_a_fn.init_from(x_act) == True: print('quan act init from the first input')
            self.init_state = 1
        if x_scale is None:
            # this is for the input quantization
            quant_act_int, act_scaling_factor = self.quan_a_fn(x_act) # TODO 应该用的是x_act，因为如果有identity需要res+x之后算一个scale给他们的!
        else:
            quant_act_int, act_scaling_factor = self.quan_a_fn(x_act) # TODO 
            if identity is not None and identity.grad is not None:
                print(identity.grad.flatten()[0].item())
            quant_act_int = FixedPointMul.apply(
                x, # int
                x_scale,
                self.activation_bit,
                act_scaling_factor,
                identity,
                identity_scaling_factor,
            )
        return quant_act_int * act_scaling_factor, act_scaling_factor, self.activation_bit # TODO

class QuantCat(nn.Module):
    """
    Quantizes the given activation.

    Args:
        activation_bit (`int`):
            Bitwidth for the quantized activation.
        act_range_momentum (`float`, *optional*, defaults to `0.95`):
            Momentum for updating the activation quantization range.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether to or not use channel-wise quantization.
        channel_len (`int`, *optional*):
            Specify the channel length when set the *per_channel* True.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(self, activation_bit=8,eval_mode=False,all_positive=False):
        super().__init__()
        self.init_state = 0
        self.activation_bit = activation_bit
        self.per_channel = False
        self.percentile = False
        self.quan_a_fn = LsqQuanParam(activation_bit, per_channel=self.per_channel, all_positive=all_positive, symmetric=True if not all_positive else False)
        self.eval_mode=eval_mode

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(activation_bit={self.activation_bit}, "
        )

    def forward(
        self,
        multi_scale_qkv,
        multi_scale_qkv_scale,
        quant_mode = False,
    ):
        if not quant_mode:
            return torch.cat(multi_scale_qkv, dim=1), None, self.activation_bit
        if not self.training and self.eval_mode: # a*scale_a=c*scale_c,out=a*(scale_a/scale_c)
            print('eval')
            x1 = multi_scale_qkv[0]
            x2 = multi_scale_qkv[1]
            x1_scale = multi_scale_qkv_scale[0]
            x2_scale = multi_scale_qkv_scale[1]
            activation_scale = self.quan_a_fn.s
            x1_out = FixedPointMulInference.apply(
                x1,
                x1_scale,
                self.activation_bit,
                activation_scale
            )
            x2_out = FixedPointMulInference.apply(
                x2,
                x2_scale,
                self.activation_bit,
                activation_scale
            )
            # print(torch.cat(multi_scale_qkv, dim=1).shape, torch.cat([x1_out, x2_out]).shape, torch.cat(multi_scale_qkv).shape)
            return torch.cat([x1_out, x2_out], dim=1), activation_scale, self.activation_bit # should use dim=1, the first dim is B
        x = torch.cat(multi_scale_qkv, dim=1) # inference的时候再用那个Scale去算
        if self.init_state == 0: # TODO: what if resume or load is perform?
            if self.quan_a_fn.init_from(x) == True: print('quan cat init from the first input')
            self.init_state = 1
        activation_q, activation_scale = self.quan_a_fn(x) # weight must be quantized at the current layer
        return activation_q * activation_scale, activation_scale, self.activation_bit


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

# register activation function here
REGISTERED_ACT_DICT: Dict[str, Type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None

# register normalization function here
REGISTERED_NORM_DICT: Dict[str, Type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name == "ln":
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None
    
    
def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def merge_tensor(x: List[torch.Tensor], mode="cat", dim=1) -> torch.Tensor:
    if mode == "cat":
        return torch.cat(x, dim=dim)
    elif mode == "add":
        return list_sum(x)
    else:
        raise NotImplementedError


def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def build_kwargs_from_config(config: Dict, target_func: Callable) -> Dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def load_state_dict_from_file(file: str, only_state_dict=True) -> Dict[str, torch.Tensor]:
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint

def list_sum(x: List) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def val2list(x: Union[List, Tuple, Any], repeat_time=1) -> List:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: Union[List, Tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> Tuple:
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

def quantize_according_to_parameters(x, scale, k):
    zero_point = torch.tensor(0.0).to(scale.device)
    n = 2 ** (k - 1) - 1
    new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
    new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
    return new_quant_x
    
class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, percentile_mode, scale):
        """
        Args:
            x (`torch.Tensor`):
                Floating point tensor to be quantized.
            k (`int`):
                Quantization bitwidth.
            percentile_mode (`bool`):
                Whether or not to use percentile calibration.
            scale (`torch.Tensor`):
                Pre-calculated scaling factor for *x*. Note that the current implementation of SymmetricQuantFunction
                requires pre-calculated scaling factor.

        Returns:
            `torch.Tensor`: Symmetric-quantized value of *input*.
        """
        zero_point = torch.tensor(0.0).to(scale.device)

        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)

        ctx.scale = scale # ctx可以在forward中保存变量，在backward的时候用
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)
        return grad_output.clone() / scale, None, None, None, None



def linear_quantize(input, scale, zero_point, inplace=False, info=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.

    Args:
        input (`torch.Tensor`):
            Single-precision input tensor to be quantized.
        scale (`torch.Tensor`):
            Scaling factor for quantization.
        zero_pint (`torch.Tensor`):
            Shift for quantization.
        inplace (`bool`, *optional*, defaults to `False`):
            Whether to compute inplace or not.

    Returns:
        `torch.Tensor`: Linearly quantized value of *input* according to *scale* and *zero_point*.
    """
    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    # quantized = float / scale + zero_point
    if inplace:
        input.mul_(1.0 / scale).add_(zero_point).round_()
        return input
    return torch.round(input / scale + zero_point)

def symmetric_linear_quantization_params(num_bits, saturation_min, saturation_max, per_channel=False):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.

    Args:
        saturation_min (`torch.Tensor`):
            Lower bound for quantization range.
        saturation_max (`torch.Tensor`):
            Upper bound for quantization range.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether to or not use channel-wise quantization.

    Returns:
        `torch.Tensor`: Scaling factor that linearly quantizes the given range between *saturation_min* and
        *saturation_max*.
    """
    # in this part, we do not need any gradient computation,
    # in order to enforce this, we put torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1

        if per_channel:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n

        else:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n

    return scale

class FixedPointMul(Function):
    """
    Function to perform fixed-point arithmetic that can match integer arithmetic on hardware.

    Args:
        pre_act (`torch.Tensor`):
            Input tensor.
        pre_act_scaling_factor (`torch.Tensor`):
            Scaling factor of the input tensor *pre_act*.
        bit_num (`int`):
            Quantization bitwidth.
        z_scaling_factor (`torch.Tensor`):
            Scaling factor of the output tensor.
        identity (`torch.Tensor`, *optional*):
            Identity tensor, if exists.
        identity_scaling_factor (`torch.Tensor`, *optional*):
            Scaling factor of the identity tensor *identity*, if exists.

    Returns:
        `torch.Tensor`: Output tensor(*pre_act* if *identity* is not given, otherwise the addition of *pre_act* and
        *identity*), whose scale is rescaled to *z_scaling_factor*.
    """

    @staticmethod
    def forward(
        ctx,
        pre_act,
        pre_act_scaling_factor, # 这里应该放比如卷积后的 s1*s2
        bit_num,
        z_scaling_factor, # 就是s3, 然后进行s1*s2/s3的操作
        identity=None, # 用到残差连接
        identity_scaling_factor=None,
    ):
        if len(pre_act_scaling_factor.shape) == 4: # 1, 8, 1, 1; input 1, 8, 480, 960
            reshape = lambda x: x  # noqa: E731
        else:
            reshape = lambda x: x.view(1, 1, -1)  # noqa: E731
       
        ctx.identity = identity

        n = 2 ** (bit_num - 1) - 1

        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            ctx.z_scaling_factor = z_scaling_factor
            z_int = torch.round(pre_act / pre_act_scaling_factor)
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            if torch.any(torch.isnan(new_scale)):
                raise ValueError("new_scale is NaN", _A, _B)
            new_scale = reshape(new_scale)
            m, e = batch_frexp(new_scale)

            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0**e))
            if identity is not None:
                # needs addition of identity activation
                wx_int = torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)
                if torch.any(torch.isnan(new_scale)):
                    raise ValueError("identity new_scale is NaN", _A, _B)

                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0**e1))

                output = output1 + output
            return torch.clamp(output.type(torch.float), -n, n-1)

    @staticmethod
    def backward(ctx, grad_output): # TODO
        if torch.any(torch.isnan(grad_output)):
            raise ValueError(grad_output, grad_output.shape)
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, identity_grad, None # 对应forward的六个参数

class FixedPointMulInference(Function):
    @staticmethod
    def forward(
        ctx,
        pre_act,
        pre_act_scaling_factor, # 这里应该放比如卷积后的 s1*s2
        bit_num,
        z_scaling_factor, # 就是s3, 然后进行s1*s2/s3的操作
        identity=None, # 用到残差连接
        identity_scaling_factor=None,
    ):
        if len(pre_act_scaling_factor.shape) == 4: # 1, 8, 1, 1; input 1, 8, 480, 960
            reshape = lambda x: x  # noqa: E731
        else:
            reshape = lambda x: x.view(1, 1, -1)  # noqa: E731
        ctx.identity = identity

        n = 2 ** (bit_num - 1) - 1

        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)
            ctx.z_scaling_factor = z_scaling_factor
            # z_int = torch.round(pre_act / pre_act_scaling_factor) originally int
            z_int = pre_act
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            new_scale = reshape(new_scale)
            m, e = batch_frexp(new_scale)
            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0**e))
            if identity is not None:
                # needs addition of identity activation
                # wx_int = torch.round(identity / identity_scaling_factor) originally int
                wx_int = identity
                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)
                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0**e1))
                output = output1 + output
            return torch.clamp(output.type(torch.float), -n, n-1)

    @staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, identity_grad, None
    
def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Args:
        scaling_factor (`torch.Tensor`):
            Target scaling factor to decompose.

    Returns:
        ``Tuple(torch.Tensor, torch.Tensor)`: mantisa and exponent
    """

    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(
            decimal.Decimal(m * (2**max_bit)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP)
        )
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = float(max_bit) - output_e

    return (
        torch.from_numpy(output_m).to(inputs.device).view(shape_of_input),
        torch.from_numpy(output_e).to(inputs.device).view(shape_of_input),
    )
    
class LsqQuanParam(nn.Module):
    def __init__(self, bit, per_channel, all_positive=False, symmetric=False, per_channel_num=None):
        super().__init__()
        self.bit = bit
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        if per_channel:
            if per_channel_num is None:
                raise NotImplementedError('per_channel_num should not be none in per_channel case')
            self.s = nn.Parameter(torch.ones(per_channel_num))
        else:
            self.s = nn.Parameter(torch.ones(1))
        self.register_buffer('initialized', torch.tensor(False))
        self.notnan = True
        self.last_s = 1.0
        self.last_grad = 1.0
        
    def init_from(self, x, *args, **kwargs):
        if not self.initialized:
            if self.per_channel:
                init_val = 2 * x.detach().abs().mean(dim=-1).mean(dim=-1).mean(dim=-1) / (self.thd_pos ** 0.5)
                self.s.data.copy_(init_val.cuda())
            else:
                init_val = 2 * x.detach().abs().mean() / (self.thd_pos ** 0.5) # sometimes very large value of x.mean()
                self.s.data.copy_(init_val.cuda())
            self.initialized = torch.tensor(True)
            return True
        else: return False


    def forward(self, x): # input x from conv layer: x shape is [NCHW], C should not be consider in per_channel mode
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5) # TODO
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(clip(self.s, torch.tensor(1e-5).float().to(x.device)), s_grad_scale)
        if self.per_channel:
            s_scale = s_scale.view(-1, 1, 1, 1)
        x = x / s_scale
        x = round_pass(x)
        x = torch.clamp(x, self.thd_neg, self.thd_pos) # TODO
        if self.s.grad is not None:
            self.last_grad = self.s.grad.item()
        elif self.s.grad is not None and self.training: print('first grad')
        if not torch.isnan(self.s): self.notnan = True
        elif torch.isnan(self.s) and self.notnan: print("suddenly nan, last_s:",self.last_s,"last_grad:",self.last_grad)
        self.last_s = self.s.item()
        return x, s_scale
    
def clip(x, eps):
    x_clip = torch.where(x > eps, x, eps)
    return x - x.detach() + x_clip.detach()

class LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, init_act_clip_val, eval_mode, dequantize=True, inplace=False):
        super(LearnedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]))
        self.dequantize = dequantize
        self.inplace = inplace
        self.eval_mode = eval_mode

    def forward(self, x, x_scale, quant_mode):
        if not quant_mode:
            return F.relu6(x), x_scale, self.num_bits
        if not self.training and self.eval_mode:
            x = F.relu(x, self.inplace) # relu
            x = torch.where(x < torch.round(self.clip_val / x_scale), x, torch.round(self.clip_val / x_scale)) # cut, like relu6 but clip_val is learnable
            print(x_scale)
            return x, x_scale, self.num_bits
        x = F.relu(x, self.inplace) # relu
        x = torch.where(x < self.clip_val, x, self.clip_val) # cut, like relu6 but clip_val is learnable
        scale, zero_point = x_scale, torch.tensor(0.0).to(x_scale.device)
        x = LinearQuantizeSTE.apply(x, scale, zero_point, self.dequantize, self.inplace) # TODO
        return x * x_scale, x_scale, self.num_bits

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val.item(),
                                                           inplace_str)

class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace)
        # if dequantize:
        #     output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None