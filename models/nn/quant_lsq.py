from audioop import bias
from re import S
import time
from sympy import N
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
import math
from threading import current_thread
import decimal
from fractions import Fraction
from decimal import Decimal
from models.nn.lsq import LsqQuantizer4input, LsqQuantizer4weight, round_pass
class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scale_sign(input):
    return ScaleSigner.apply(input)


class Quantizer(Function):
    @staticmethod
    def forward(ctx, input, nbit, type="dorefa"):
        scale = 2 ** nbit - 1
        if type == "acy":
            scale = 2 ** (nbit - 1)
            
        output = torch.round(input * scale)
        y = torch.clamp(output, min = -scale, max = scale - 1)
        
        # print ("quantized-----------", nbit, scale, output.flatten()[:10])
        # print ("quantized-----------", nbit, scale, y.flatten()[:10])
        
        # if len(input.shape) == 2 and input.shape[1] == 512:
        #     print (y[3,:].flatten()[:10], torch.min(y[3,:].flatten()), torch.max(y[3,:].flatten()))

        return y / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def quantize(input, nbit, type="dorefa"):
    return Quantizer.apply(input, nbit, type)


def dorefa_w(w, nbit_w):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = quantize(w, nbit_w) - 0.5
    return w


def dorefa_a(input, nbit_a):
    return quantize(torch.clamp(0.125 * input, 0, 1), nbit_a)

def acy_w(w, nbit_w, pre_max=-1, is_train=True, is_round=1):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        # w = torch.tanh(w)
        max_w = pre_max
        if pre_max < 0:
            max_w = torch.max(torch.abs(w))
            max_w = torch.pow(2, torch.ceil(torch.log2(max_w)))
        if max_w == 0:
            return w , max_w

        if True:  # is_train:
            w = w / max_w
            w = torch.clip(w, -1, 1)
            w = max_w * quantize(w, nbit_w, "acy")
            # w = w / (2 * max_w) + 0.5
            # w = 2 * max_w * (quantize(w, nbit_w) - 0.5)
        else:
            ub = 2 ** (nbit_w - 1) - 1
            lb = -2 ** (nbit_w - 1)
            q = (-lb) / max_w
            w = torch.round(w * q)
            w = torch.clamp(w, lb, ub) / q
            print(max_w)
        #####################
        # if (w.dim() == 4):
        #     w = torch.zeros(w.shape)
    return w, max_w


def acy_a(input, nbit_a, pre_max=-1, is_train=True):
    if nbit_a == 1:
        a = scale_sign(input)
    else:
        max_a = torch.max(torch.abs(input))
        pow = torch.ceil(torch.log2(max_a))
        max_a = torch.pow(2, pow)
        if max_a == 0:
            return input
        a = input / max_a
        a = torch.clip(a, -1, 1)
        a = max_a * quantize(a, nbit_a, "acy")
        err = 1
        last_mse = torch.abs(torch.norm(a - input))
        while err > 0:
            pow = pow - 1
            max_a = torch.pow(2, pow)
            a = input / max_a
            a = torch.clip(a, -1, 1)
            a = max_a * quantize(a, nbit_a, "acy")
            err = last_mse
            last_mse = torch.abs(torch.norm(a - input))
            err = err - last_mse
        max_a = torch.pow(2, pow + 1)
        if True:  # is_train:
            a = input / max_a
            a = torch.clip(a, -1, 1)
            a = max_a * quantize(a, nbit_a, "acy")
            # a = torch.clip(input / (2 * max_a) + 0.5,0,1)
            # a = 2 * max_a * (quantize(a, nbit_a) - 0.5)
        else:
            ub = 2 ** (nbit_a - 1) - 1
            lb = -2 ** (nbit_a - 1)
            q = (-lb) / max_a
            a = torch.round(input * q)
            a = torch.clamp(a, lb, ub) / q

    return a, max_a



'''
def dorefa_a(input, nbit_a):
    #PACT Activation modify
    x = F.relu(input)
    clip_val = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
    clip_val.to('cuda')
    x = torch.where(x < clip_val, x, clip_val)
    n = float(2 ** nbit_a - 1) / nbit_a
    x_forward = torch.round(x * n) / n
    out = x_forward + x - x.detach()
    return out
'''


def batch_frexp(inputs, bit=8):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """
    shape_of_input = inputs.size()

    # transform the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())

    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2 ** bit)).quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = 1.0*bit - output_e

    return torch.from_numpy(output_m).to(inputs.device).view(shape_of_input), \
           torch.from_numpy(output_e).to(inputs.device).view(shape_of_input)


class PActFn(Function):
    @staticmethod
    def forward(ctx, x, alpha, k): # k=8
        # if alpha <= 0:
        #     print ("org scale:", alpha, alpha.grad)
        ctx.save_for_backward(x, alpha)
        pow = torch.ceil(torch.log2(alpha**2)) 
        clip_val = torch.pow(2, pow)
    # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        scale = (2**(k - 1)) / clip_val
        y = torch.clamp(x + torch.sign(x) * 1e-6, min = -clip_val.item(), max = ((2**(k - 1) - 1) / scale).item())
        y_q = torch.trunc(y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py
        # We get dL / dy_q as a gradient
        # print(current_thread())
        x, alpha, = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha]
        # Actual gradient for alpha,
        # By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
        # dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range
        dldy_sum = torch.sum(dLdy_q)
        # if dldy_sum.item() != dldy_sum.item():
        #     dLdy_q = torch.zeros(dLdy_q.shape).to(dLdy_q.device)
        #     print ("PACT grad error 1", dLdy_q.shape)
        #     print (dLdy_q)
        #     print ("###############################################")
        #     print (alpha)
        #     print ("###############################################")
        #     print (x)
        #     exit()

        pow = torch.ceil(torch.log2(alpha**2))
        clip_val = torch.pow(2, pow)
        lower_bound = x < -clip_val
        upper_bound = x > clip_val
        # x_range       = 1.0-lower_bound-upper_bound
        x_range = ~(lower_bound |upper_bound)
        grad_alpha = 2 * alpha * torch.sum(dLdy_q * (torch.ge(x, clip_val) | torch.le(x, -clip_val)).float()).view(-1)
        # if grad_alpha.item() != grad_alpha.item():
        #     print ("PACT grad error 2", x.shape)
        #     print (alpha)
        #     print ("###############################################")
        #     print (x)
        #     exit()
        # x_range_sum = torch.sum(x_range.float())
        # if x_range_sum.item() != x_range_sum.item():
        #     print ("PACT grad error 3")
        #     exit()

        # if alpha - grad_alpha <= 0:
        #     grad_alpha = torch.zeros(grad_alpha.shape).to(grad_alpha.device)
        return dLdy_q * x_range.float(), grad_alpha, None


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize floating point input tensor to integers with the given scaling factor and zeropoint.

    Parameters:
    ----------
    input: floating point input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    # reshape scale and zeropoint for convolutional weights and activations
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
    if inplace:
        input.mul_(1. / scale).add_(zero_point).round_()
        return input
    return torch.round(1. / scale * input + zero_point)

class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        n = 2 ** (k - 1) - 1

        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

        zero_point = torch.tensor(0.).to(x.device)

        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)

        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)

        ctx.scale = scale
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

        return grad_output.clone() / scale, None, None, None


class PACT(nn.Module):
    def __init__(self, num_bits):
        super(PACT, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([2]), requires_grad=True)

    def forward(self, x):

        # pow = torch.ceil(torch.log2(self.clip_val))
        # self.clip_val[0] = torch.pow(2, pow)[0]
        # x = F.relu(x)
        # x = torch.where(x < self.clip_val, x, self.clip_val)
        # x = torch.where(x < self.clip_val, x, self.clip_val)
        # x = torch.where(x > -self.clip_val, x, -self.clip_val)
        # n = float(2 ** (self.num_bits - 1)) / self.clip_val
        # x_forward = torch.round(x * n) / n
        # out = x_forward + x - x.detach()
        out = PActFn.apply(x, self.clip_val, self.num_bits)
        pow = torch.ceil(torch.log2(self.clip_val**2))
        max_out = torch.pow(2, pow)[0]
        scale_a = max_out / float((2**(self.num_bits - 1)))
        return out, scale_a

global_idx = 0

def get_next_global_idx():
    global global_idx
    global_idx = global_idx + 1
    return global_idx

def reset_global_idx():
    global global_idx
    global_idx = 0
    
def get_global_idx():
    global global_idx
    return global_idx

def symmetric_linear_quantization_params(num_bits,
                                         saturation_min,
                                         saturation_max,
                                         per_channel=False):
    """
    Compute the scaling factor and zeropoint with the given quantization range for symmetric quantization.

    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    per_channel: if True, calculate the scaling factor per channel.
    """

    # these computation do not require any gradients, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        scale1, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
        if per_channel:
            # scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale1, min=1e-8) / n
        else:
            scale = torch.max(scale1)
            scale = torch.ones(scale1.shape).to(saturation_min.device) * torch.clamp(scale, min=1e-8) / n
    if torch.sum(torch.isnan(scale.flatten())) == 0:
        scale_m, scale_e = batch_frexp(scale, bit=8)
        scale = (scale_m / torch.pow(2, scale_e)).type(torch.float32)
    return scale

# class QuanConv(nn.Conv2d):
#     """docstring for QuanConv"""

#     def __init__(self, in_channels, out_channels, kernel_size, quan_name_w='lsq', quan_name_a='acy',
#                  nbit_w=8, nbit_a=8, stride=1, padding=0, norm='bn2d', per_channel=True,
#                  dilation=1, groups=1, bias=True, psum_quan=False, cg=32):
#         super(QuanConv, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             groups, bias)
#         self.fixBN = True
#         self.psum_quan = psum_quan
#         self.cg = cg
#         self.per_channel = per_channel
#         self.all_positive = False
#         self.log_tensor = False
#         self.nbit_w = nbit_w
#         self.nbit_a = nbit_a
#         self.norm = norm
#         self.w = torch.zeros(self.weight.shape)
#         self.b = torch.zeros(self.weight.shape[0])
#         self.scale  = [0.0, [0.0]*self.w.shape[0], 0.0]
#         self.lsq_w  = LsqQuantizer4weight(
#                         bit=self.nbit_w,
#                         all_positive=self.all_positive,
#                         per_channel=self.per_channel,
#                         per_channel_num=self.weight.shape[0])
#         # self.lsq_w.init_from(self.weight)
#         self.lsq_a  = LsqQuantizer4input(
#                         bit=self.nbit_a,
#                         all_positive=self.all_positive,
#                         per_channel=False)
#         self.statsq = StatsQuantizer(
#                         num_bits=self.nbit_w,  
#                         per_channel=self.per_channel)
#         if self.psum_quan:
#             self.lsq_p = []
#             for i in range(self.weight.shape[1] // cg):
#                 self.lsq_p.append(LsqQuantizer4input(
#                                     bit=8,
#                                     all_positive=self.all_positive,
#                                     per_channel=False)
#                                 )
#             self.loss_type = nn.MSELoss()
#             self.loss_weight = 1
#             self.psum_loss = 0.0
#         name_w_dict = {'dorefa': dorefa_w, "acy": acy_w, "lsq": self.lsq_w, "statsq": self.statsq}
#         name_a_dict = {'dorefa': dorefa_a, "acy": acy_a, "lsq": self.lsq_a}
#         self.quan_w = name_w_dict[quan_name_w]
#         self.quan_a = name_a_dict[quan_name_a]
#         self.quan_a_name = quan_name_a
#         self.a_max = torch.ones(1, requires_grad=False)
#         self.scale_a = torch.ones(1, requires_grad=False)

#         self.bn_weight = torch.nn.parameter.Parameter(torch.ones(out_channels))
#         self.bn_bias = torch.nn.parameter.Parameter(torch.zeros(out_channels))

#         self.register_buffer('running_mean', torch.zeros(out_channels))
#         self.register_buffer('running_var', torch.ones(out_channels))
#         # self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         self.running_mean = torch.zeros(out_channels)
#         self.running_var = torch.ones(out_channels)
#         self.momentum = 0.1
#         self.clip_val = torch.nn.Parameter(torch.Tensor([1]))
#         if self.per_channel:
#             self.scale_w = torch.ones(self.weight.shape[0],1,1,1, requires_grad=False)
#         else:
#             self.scale_w = torch.ones(1,1,1,1, requires_grad=False)

#     def set_fix(self, setbn):
#         self.fixBN = setbn
#     def set_loss(self, setloss):
#         self.psum_loss = setloss
#     def set_per_channel(self, set_per_ch):
#         self.per_channel = set_per_ch
#     def set_all_positive(self, set_all_pos):
#         self.per_channel = set_all_pos
#     def set_log_tensor(self, set_log_tensor):
#         self.log_tensor = set_log_tensor

#     def psum_quantizer(self, input, scale_a, weight, scale_w, bias, c_g):
#         B, C, H, W = input.shape
#         C_out = weight.shape[0]
#         # initialize partial sums
#         acc_psums = torch.zeros((B, C_out, H, W), dtype=input.dtype).to(input.device)
#         acc_psums_unquan = torch.zeros((B, C_out, H, W), dtype=input.dtype).to(input.device)
#         for i, c_in in enumerate(range(0, C, c_g)):
#             psum = F.conv2d(input=input[:, c_in:c_in+c_g, :, :], 
#                                    weight=weight[:, c_in:c_in+c_g, :, :],
#                                    bias=None,
#                                    stride=self.stride,
#                                    padding=self.padding,
#                                    dilation=self.dilation,
#                                    groups=self.groups).to(input.device)
#             if c_in < C - c_g:
#                 acc_psums_unquan += psum
#                 acc_psums, _ = self.lsq_p[i](acc_psums + psum)

#         acc_psums = acc_psums + psum
#         output = acc_psums + bias.view(1, C_out, 1, 1)
#         return output * scale_w * scale_a

#     # @weak_script_method
#     def forward(self, input1):
#         global global_idx
#         # quantize input
#         self.scale_a = self.a_max
#         x = input1 / self.scale_a
        
#         if self.training:
#             with torch.no_grad():
#                 if not self.per_channel:
#                     mean_bn = self.running_mean
#                     var_bn = self.running_var
#                     tmp = self.bn_weight / torch.sqrt(var_bn + 1e-5)
#                     tmp1 = self.bn_bias        
                        
#                     w = tmp.view(tmp.size()[0], 1, 1, 1) * self.weight
#                     w_transform = w.data.contiguous().view(self.out_channels, -1)
#                     w_min = w_transform.min(dim=1).values
#                     w_max = w_transform.max(dim=1).values
#                     weight_scaling_factor = symmetric_linear_quantization_params(self.nbit_w, w_min, w_max, False) / tmp.view(tmp.size()[0])
                
#                 else:
#                     w_transform = self.weight.data.contiguous().view(self.out_channels, -1)
#                     w_min = w_transform.min(dim=1).values
#                     w_max = w_transform.max(dim=1).values
#                     weight_scaling_factor = symmetric_linear_quantization_params(self.nbit_w, w_min, w_max, True)
            
#             weight_integer = SymmetricQuantFunction.apply(self.weight, self.nbit_w, weight_scaling_factor)
#             if self.bias != None: 
#                 bias_integer = SymmetricQuantFunction.apply(self.bias, 16, weight_scaling_factor * self.scale_a)   
#                 output = F.conv2d(x, weight_integer, bias_integer, self.stride, self.padding, self.dilation, self.groups)
#             else:
#                 output = F.conv2d(x, weight_integer, self.bias, self.stride, self.padding, self.dilation, self.groups)
            
#             output = output * weight_scaling_factor.view(1, -1, 1, 1) * self.scale_a
            
#             output = F.batch_norm(output, self.running_mean, self.running_var, weight=self.bn_weight,
#                                 bias=self.bn_bias, training=self.training, momentum=0.1, eps=1e-05)
#             return output
            
        
#         mean_bn = self.running_mean
#         var_bn = self.running_var
            
#         tmp = self.bn_weight / torch.sqrt(var_bn + 1e-5)
#         tmp1 = self.bn_bias
            
#         w = tmp.view(tmp.size()[0], 1, 1, 1) * self.weight

#         if self.bias != None:
#             b = tmp*(self.bias - mean_bn) + tmp1
#         else:
#             b = tmp*(0 - mean_bn) + tmp1

#         w_transform = w.data.contiguous().view(self.out_channels, -1) # shape: (out_channels, -1)
#         w_min = w_transform.min(dim=1).values # shape: (out_channels,)
#         w_max = w_transform.max(dim=1).values # shape: (out_channels,)
#         weight_scaling_factor = symmetric_linear_quantization_params(self.nbit_w, w_min, w_max, self.per_channel)
        
#         weight_integer = SymmetricQuantFunction.apply(w, self.nbit_w, weight_scaling_factor)
#         bias_integer = SymmetricQuantFunction.apply(b, 16, weight_scaling_factor * self.scale_a)   
        
#         if not self.training and self.log_tensor:
#             self.w.copy_(weight_integer * weight_scaling_factor.view(-1,1,1,1))
#             self.b.copy_(bias_integer * weight_scaling_factor * self.scale_a)
#             self.scale[0] = 1.0 / torch.max(weight_scaling_factor)
#             self.scale[1] = self.scale_a
            
            
#         output2 = F.conv2d(x, weight_integer, bias_integer, self.stride, self.padding,
#                            self.dilation, self.groups) * weight_scaling_factor.view(1,-1,1,1) * self.scale_a
             
#         return output2


class QuanConv(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w='lsq', quan_name_a='acy',
                 nbit_w=8, nbit_a=8, stride=1, padding=0, norm='bn2d', per_channel=True,
                 dilation=1, groups=1, bias=True, psum_quan=False, cg=32):
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.fixBN = True
        self.psum_quan = psum_quan
        self.cg = cg
        self.per_channel = per_channel
        self.all_positive = False
        self.log_tensor = False
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.norm = norm
        self.w = torch.zeros(self.weight.shape)
        self.b = torch.zeros(self.weight.shape[0])
        self.scale  = [0.0, [0.0]*self.w.shape[0], 0.0]
        self.lsq_w  = LsqQuantizer4weight(
                        bit=self.nbit_w,
                        all_positive=self.all_positive,
                        per_channel=self.per_channel,
                        per_channel_num=self.weight.shape[0])
        # self.lsq_w.init_from(self.weight)
        self.lsq_a  = LsqQuantizer4input(
                        bit=self.nbit_a,
                        all_positive=self.all_positive,
                        per_channel=False)
        if self.psum_quan:
            self.lsq_p = []
            for i in range(self.weight.shape[1] // cg):
                self.lsq_p.append(LsqQuantizer4input(
                                    bit=8,
                                    all_positive=self.all_positive,
                                    per_channel=False)
                                )
            self.loss_type = nn.MSELoss()
            self.loss_weight = 1
            self.psum_loss = 0.0
        name_w_dict = {'dorefa': dorefa_w, "acy": acy_w, "lsq": self.lsq_w}
        name_a_dict = {'dorefa': dorefa_a, "acy": acy_a, "lsq": self.lsq_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]
        self.quan_a_name = quan_name_a
        self.a_max = torch.ones(1, requires_grad=False)
        self.scale_a = torch.ones(1, requires_grad=False)

        self.bn_weight = torch.nn.parameter.Parameter(torch.ones(out_channels))
        self.bn_bias = torch.nn.parameter.Parameter(torch.zeros(out_channels))

        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        # self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.running_mean = torch.zeros(out_channels)
        self.running_var = torch.ones(out_channels)
        self.momentum = 0.1
        if self.per_channel:
            self.scale_w = torch.ones(self.weight.shape[0],1,1,1, requires_grad=False)
        else:
            self.scale_w = torch.ones(1,1,1,1, requires_grad=False)

    def set_fix(self, setbn):
        self.fixBN = setbn
    def set_loss(self, setloss):
        self.psum_loss = setloss
    def set_per_channel(self, set_per_ch):
        self.per_channel = set_per_ch
    def set_all_positive(self, set_all_pos):
        self.per_channel = set_all_pos
    def set_log_tensor(self, set_log_tensor):
        self.log_tensor = set_log_tensor

    def psum_quantizer(self, input, scale_a, weight, scale_w, bias, c_g):
        B, C, H, W = input.shape
        C_out = weight.shape[0]
        # initialize partial sums
        acc_psums = torch.zeros((B, C_out, H, W), dtype=input.dtype).to(input.device)
        acc_psums_unquan = torch.zeros((B, C_out, H, W), dtype=input.dtype).to(input.device)
        for i, c_in in enumerate(range(0, C, c_g)):
            psum = F.conv2d(input=input[:, c_in:c_in+c_g, :, :], 
                                   weight=weight[:, c_in:c_in+c_g, :, :],
                                   bias=None,
                                   stride=self.stride,
                                   padding=self.padding,
                                   dilation=self.dilation,
                                   groups=self.groups).to(input.device)
            if c_in < C - c_g:
                acc_psums_unquan += psum
                acc_psums, _ = self.lsq_p[i](acc_psums + psum)

        acc_psums = acc_psums + psum
        output = acc_psums + bias.view(1, C_out, 1, 1)
        return output * scale_w * scale_a

    # @weak_script_method
    def forward(self, input1):
        global global_idx
        # quantize input
        x = input1  
        if self.training and self.norm != None:
            output = F.conv2d(x , self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
            mean_bn = output.mean([0, 2,3], keepdim=False).squeeze(0)
            var_bn = output.var([0, 2,3], keepdim=False).squeeze(0)
            output1 = F.batch_norm(output, self.running_mean, self.running_var, weight=self.bn_weight,
                                   bias=self.bn_bias, training=self.training, momentum=0.1, eps=1e-05)
        if not self.training:
            mean_bn = self.running_mean
            var_bn = self.running_var
            
        if self.norm != None:
            tmp = self.bn_weight / torch.sqrt(var_bn + 1e-5)
            w = tmp.view(tmp.size()[0], 1, 1, 1) * self.weight
            b = self.bias
            if self.bias != None:
                b = tmp*(self.bias - mean_bn) + self.bn_bias
            else:
                b = tmp*(0 - mean_bn) + self.bn_bias
        else:
            w = self.weight
            b = self.bias

        weight_integer, weight_scaling_factor = self.quan_w(w)
        weight_integer = weight_integer.to(x.device)
        weight_scaling_factor = weight_scaling_factor.to(x.device)
        self.scale_a = self.scale_a.to(x.device)
        # print(self.scale_a)
        # print(weight_scaling_factor) 
        if b != None:
            b = b.to(x.device)
            bias_integer = SymmetricQuantFunction.apply(b, 16, self.scale_a * weight_scaling_factor.squeeze()) * self.scale_a * weight_scaling_factor.squeeze()
            bias_integer = bias_integer.to(x.device)
        else:
            bias_integer = None

        if get_global_idx() == 0:
            np.save("npz_logging/input.npy",
                (x/self.scale_a).detach().cpu().numpy()
                )  
        output2 = F.conv2d(x, weight_integer, bias_integer, self.stride, self.padding, self.dilation, self.groups) 


        if self.training and self.norm != None:
            # output1.grad = output2.grad
            output1 = output1 - output2.detach() + output2

        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()
            if bias_integer != None:
                np.savez("npz_logging/" + str(idx) + "_conv",
                         w=weight_integer.detach().cpu().numpy(),
                         b=bias_integer.detach().cpu().numpy(), 
                         input_scale=self.scale_a.detach().cpu().numpy(), 
                         weight_scales=weight_scaling_factor.detach().cpu().numpy(), 
                         input=x.detach().cpu().numpy(), 
                         output=output2.detach().cpu().numpy()
                         )
            else:
                np.savez("npz_logging/" + str(idx) + "_conv",
                         w=weight_integer.detach().cpu().numpy(), 
                         input_scale=self.scale_a.detach().cpu().numpy(), 
                         weight_scales=weight_scaling_factor.detach().cpu().numpy(), 
                         input=x.detach().cpu().numpy(), 
                         output=output2.detach().cpu().numpy()
                        )
        return output2


class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=1, nbit_a=1):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.scale = [0.0, 0.0,0.0]
        name_w_dict = {'dorefa': dorefa_w, "acy": acy_w}
        name_a_dict = {'dorefa': dorefa_a, "acy": acy_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]
        self.w = torch.zeros(self.weight.shape)
        self.b = torch.zeros(self.weight.shape[0])
        self.clip_val = torch.nn.Parameter(torch.Tensor([2.0]))
        self.per_channel = True
        self.log_tensor = False
        
    def set_per_channel(self, set_per_ch):
        self.per_channel = set_per_ch
    def set_log_tensor(self, set_log_tensor):
        self.log_tensor = set_log_tensor

    # @weak_script_method
    def forward(self, input1):
        global global_idx
        input2 = PActFn.apply(input1, self.clip_val, self.nbit_a)
        pow = torch.ceil(torch.log2(self.clip_val**2))
        max_a = torch.pow(2, pow)[0]      
        x = input2 * (2**(self.nbit_a - 1)) / max_a
        
        
        w = self.weight
        w_transform = w.data.detach()
        w_min, _ = torch.min(w_transform, dim=1, out=None)
        w_max, _ = torch.max(w_transform, dim=1, out=None)
            
        weight_scaling_factor = symmetric_linear_quantization_params(self.nbit_w, w_min, w_max, self.per_channel)
        
        weight_integer = SymmetricQuantFunction.apply(self.weight, self.nbit_w, weight_scaling_factor)
        if self.bias != None: 
            bias_integer = SymmetricQuantFunction.apply(self.bias, 16, weight_scaling_factor * max_a / (2**(self.nbit_a - 1)))   
            output = F.linear(x, weight_integer, bias_integer)
        else: 
            output = F.linear(x, weight_integer, self.bias)
        
        output = output * weight_scaling_factor.view(1, -1) * max_a / (2**(self.nbit_a - 1))

        if not self.training and self.log_tensor:
            self.w.copy_(weight_integer * weight_scaling_factor.view(-1, 1))
            self.b.copy_(bias_integer * weight_scaling_factor * max_a / (2**(self.nbit_a - 1)))
            self.scale[0] = 1.0 / torch.max(weight_scaling_factor)
            self.scale[1] = (2**(self.nbit_a - 1)) / max_a
            
        if not self.training and self.log_tensor and self.nbit_a < 32:
            _, max_out = self.quan_a(
                output, self.nbit_a, is_train=self.training)
            self.scale[2] = (2**(self.nbit_a - 1)) / max_out

        return output
