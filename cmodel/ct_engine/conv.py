from math import e
from re import A
import re
from typing import Dict, List, Tuple, Union, Optional
from sympy import per

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_same_padding, resize, val2list, val2tuple, merge_tensor
from models.nn.norm import build_norm
from models.nn.act import build_act, Quanhswish
from models.nn.quant_lsq import QuanConv, PActFn, PACT, SymmetricQuantFunction
from models.nn.lsq import LsqQuantizer4input, LsqQuantizer4weight
from models.nn.ops import ConvLayer, DSConv, MBConv, EfficientViTBlock, OpSequential, ResidualBlock, IdentityLayer, LiteMSA
__all__ = [
    "QConvLayer",
    "QDSConv",
    "QMBConv",
    "QLiteMSA",
    "QEfficientViTBlock",
    "QResidualBlock",
    # "DAGBlock",
    # "OpSequential",
]


#################################################################################
#                             Basic Layers                                      #
#################################################################################
class Conv(nn.Conv2d):
    """docstring for QuanConv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, per_channel=True,
                 dilation=1, groups=1, bias=True, input_bitdepth=8, weight_bitdepth=8, output_bitdepth=8):
        super(Conv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.fixBN = True
        self.per_channel = per_channel
        # scale
        self.input_scale = torch.ones(1, requires_grad=False) # 2^N
        # reciprocal format (c / 2^N)
        self.weight_scale = torch.ones(1, requires_grad=False) if self.per_channel \
                       else torch.ones(self.weight.shape[0], requires_grad=False)
        self.output_scale = torch.ones(1, requires_grad=False)
        # bit-width
        self.input_bitdepth = input_bitdepth
        self.weight_bitdepth = weight_bitdepth
        self.output_bitdepth = output_bitdepth
        # clamp value
        self.thd_neg = -2 ** (self.output_bitdepth - 1)
        self.thd_pos = 2 ** (self.output_bitdepth - 1) - 1
    def forward(self, input): # the value of input is int8 but the type is float32
        # weight 
        w = self.weight / self.weight_scale # stored as Sw*W_int in fp format
        # bias
        b = self.bias / (self.weight_scale * self.input_scale) if self.bias is not None else None
        # perform convolution in fp format to simulate integer convolution
        output = F.conv2d(input, w, b,
                          self.stride, self.padding, self.dilation, self.groups)
        # dyadic scale
        dyadic_scale = self.input_scale * self.weight_scale / self.output_scale
        # final output
        output = output * dyadic_scale 
        # clamp
        output = torch.clamp(output, self.thd_neg, self.thd_pos)
        return output
    
class Conv(nn.Conv2d):
    """docstring for QuanConv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True,  output_bitdepth=8):
        super(Conv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        # bit-width
        self.output_bitdepth = output_bitdepth
        # clamp value
        self.thd_neg = -2 ** (self.output_bitdepth - 1)
        self.thd_pos = 2 ** (self.output_bitdepth - 1) - 1
        
    def forward(self, input, weight, bias, coeff, act_func: str): 
        # MAC-array
        output = F.conv2d(input, weight, None,
                          self.stride, self.padding, self.dilation, self.groups)
        # ADD
        output = output + bias if bias is not None else output
        # dyadic
        output = output * coeff 
        output = torch.clamp(output, self.thd_neg, self.thd_pos)
        # ACT
        act = build_act(act_func)
        output = act(output)
        return output

