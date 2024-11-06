import torch 
from torch.autograd import Variable
import torch.nn as nn 
from torch.autograd import Function 
from torch.nn.parameter import Parameter 
from torch import optim 
import torch.nn.functional as F 
from torchvision import datasets, transforms 
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def activation(x, scale, type):
    device = x.device
    gelu_seg_point_scale = torch.tensor([-3, -1, -0.5, 0, 0.5, 1, 3],dtype=torch.float32, device=device)
    gelu_slope = torch.tensor([0.0, -0.078125, 0.015625, 0.3125, 0.6875, 0.984375, 1.078125, 1.0],dtype=torch.float32, device=device) 
    gelu_intercept = torch.tensor([0.0, -0.234375, -0.15625, 0.0, 0.0, -0.15625, -0.234375, 0.0],dtype=torch.float32, device=device) 

    swish_seg_point_scale = torch.tensor([-3, -1, -0.5, 0, 0.5, 1, 3],dtype=torch.float32, device=device)
    swish_slope = torch.tensor([0.0, -0.140625, 0.15625, 0.375, 0.625, 0.84375, 1.140625, 1.0],dtype=torch.float32, device=device) 
    swish_intercept = torch.tensor([0.0, -0.40625, -0.109375, 0.0, 0.0, -0.109375, -0.40625, 0.0],dtype=torch.float32, device=device) 

    sigmoid_seg_point_scale = torch.tensor([-3, -2, -1, 0, 1, 2, 3],dtype=torch.float32, device=device)
    sigmoid_slope = torch.tensor([0.0, 0.125, 0.15625, 0.234375, 0.234375, 0.15625, 0.125, 0.0],dtype=torch.float32, device=device) 
    sigmoid_intercept = torch.tensor([0.0, 0.359375, 0.421875, 0.5, 0.5, 0.578125, 0.640625, 1.0],dtype=torch.float32, device=device) 

    tanh_seg_point_scale = torch.tensor([-3, -1, -0.5, 0, 0.5, 1, 3],dtype=torch.float32, device=device)
    tanh_slope = torch.tensor([0.0, 0.125, 0.59375, 0.921875, 0.921875, 0.59375, 0.125, 0.0],dtype=torch.float32, device=device) 
    tanh_intercept = torch.tensor([-1.0, -0.640625, -0.15625, 0.0, 0.0, 0.15625, 0.640625, 1.0],dtype=torch.float32, device=device) 

    if type == 'gelu':
        slope = gelu_slope 
        intercept = gelu_intercept
        seg_point = gelu_seg_point_scale
    elif type == 'swish':
        slope = swish_slope
        intercept = swish_intercept
        seg_point = swish_seg_point_scale
    elif type == 'sigmoid':
        slope = sigmoid_slope
        intercept = sigmoid_intercept
        seg_point = sigmoid_seg_point_scale
    elif type == 'tanh':
        slope = tanh_slope
        intercept = tanh_intercept
        seg_point = tanh_seg_point_scale

    seg_point_scale = seg_point / scale
    intercept_scale = intercept / scale
    # y = (y - y_grad).detach() + y_grad
    # we set (y - y_grad) = y_detc
    y_detc = torch.zeros_like(x,requires_grad=False)
    y_grad = x.clone()

    y_detc[x.lt(seg_point_scale[0])]                           = intercept_scale[0] 
    y_detc[x.ge(seg_point_scale[0])&x.lt(seg_point_scale[1])]  = intercept_scale[1]
    y_detc[x.ge(seg_point_scale[1])&x.lt(seg_point_scale[2])]  = intercept_scale[2]
    y_detc[x.ge(seg_point_scale[2])&x.lt(seg_point_scale[3])]  = intercept_scale[3]
    y_detc[x.ge(seg_point_scale[3])&x.lt(seg_point_scale[4])]  = intercept_scale[4]
    y_detc[x.ge(seg_point_scale[4])&x.lt(seg_point_scale[5])]  = intercept_scale[5]
    y_detc[x.ge(seg_point_scale[5])&x.lt(seg_point_scale[6])]  = intercept_scale[6]
    y_detc[x.ge(seg_point_scale[6])]                           = intercept_scale[7]

    y_grad[x.lt(seg_point_scale[0])]                           = slope[0]*y_grad[x.lt(seg_point_scale[0])]
    y_grad[x.ge(seg_point_scale[0])&x.lt(seg_point_scale[1])]  = slope[1]*y_grad[x.ge(seg_point_scale[0])&x.lt(seg_point_scale[1])] 
    y_grad[x.ge(seg_point_scale[1])&x.lt(seg_point_scale[2])]  = slope[2]*y_grad[x.ge(seg_point_scale[1])&x.lt(seg_point_scale[2])]
    y_grad[x.ge(seg_point_scale[2])&x.lt(seg_point_scale[3])]  = slope[3]*y_grad[x.ge(seg_point_scale[2])&x.lt(seg_point_scale[3])] 
    y_grad[x.ge(seg_point_scale[3])&x.lt(seg_point_scale[4])]  = slope[4]*y_grad[x.ge(seg_point_scale[3])&x.lt(seg_point_scale[4])] 
    y_grad[x.ge(seg_point_scale[4])&x.lt(seg_point_scale[5])]  = slope[5]*y_grad[x.ge(seg_point_scale[4])&x.lt(seg_point_scale[5])]
    y_grad[x.ge(seg_point_scale[5])&x.lt(seg_point_scale[6])]  = slope[6]*y_grad[x.ge(seg_point_scale[5])&x.lt(seg_point_scale[6])] 
    y_grad[x.ge(seg_point_scale[6])]                           = slope[7]*y_grad[x.ge(seg_point_scale[6])] 

    return  y_detc.detach() + y_grad 


class hardpwl(nn.Module):
    def __init__(self, pwl_type:str) -> None:
        super(hardpwl, self).__init__()
        # fp func
        self.act_funcs = {
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        # pwl func
        self.pwl_funcs = {
            'gelu':{
                'slope': torch.tensor([0.0, -0.078125, 0.015625, 0.3125, 0.6875, 0.984375, 1.078125, 1.0],dtype=torch.float32), 
                'intercept': torch.tensor([0.0, -0.234375, -0.15625, 0.0, 0.0, -0.15625, -0.234375, 0.0],dtype=torch.float32),
                'seg_point': torch.tensor([-3, -1, -0.5, 0, 0.5, 1, 3],dtype=torch.float32)
            },
            'swish':{
                'slope': torch.tensor([0.0, -0.140625, 0.15625, 0.375, 0.625, 0.84375, 1.140625, 1.0],dtype=torch.float32),
                'intercept': torch.tensor([0.0, -0.40625, -0.109375, 0.0, 0.0, -0.109375, -0.40625, 0.0],dtype=torch.float32),
                'seg_point': torch.tensor([-3, -1, -0.5, 0, 0.5, 1, 3],dtype=torch.float32)
            },
            'sigmoid':{
                'slope': torch.tensor([0.0, 0.125, 0.15625, 0.234375, 0.234375, 0.15625, 0.125, 0.0],dtype=torch.float32),
                'intercept': torch.tensor([0.0, 0.359375, 0.421875, 0.5, 0.5, 0.578125, 0.640625, 1.0],dtype=torch.float32),
                'seg_point': torch.tensor([-3, -2, -1, 0, 1, 2, 3],dtype=torch.float32)
            },
            'tanh':{
                'slope': torch.tensor([0.0, 0.125, 0.59375, 0.921875, 0.921875, 0.59375, 0.125, 0.0],dtype=torch.float32),
                'intercept': torch.tensor([-1.0, -0.640625, -0.15625, 0.0, 0.0, 0.15625, 0.640625, 1.0],dtype=torch.float32),
                'seg_point': torch.tensor([-3, -1, -0.5, 0, 0.5, 1, 3],dtype=torch.float32)
            }
        }
        self.pwl_type = pwl_type
        self.scale = torch.ones(1, requires_grad=False)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        # Get the correct activation parameters and function
        params = self.pwl_funcs[self.pwl_type]
        func = self.act_funcs[self.pwl_type]
        seg_point_scale = params['seg_point'].to(device) / self.scale
        intercept_scale = params['intercept'].to(device) / self.scale
        # pwl func
        pwl_func = torch.zeros_like(input).to(device)
        for i in range(len(seg_point_scale)):
            if i == 0:
                mask = input.lt(seg_point_scale[i])
            elif i == len(seg_point_scale) - 1:
                mask = input.ge(seg_point_scale[i-1])
            else:
                mask = input.ge(seg_point_scale[i-1]) & input.lt(seg_point_scale[i])
            pwl_func = torch.where(mask, intercept_scale[i] + params['slope'][i] * input, pwl_func)
        return (pwl_func - func(input)).detach() + func(input)
        
        
if __name__ == '__main__':

    scale = torch.Tensor([0.1]).cuda()
    test_x = torch.ones([1, 1, 128, 128], requires_grad=True, device='cuda')
    
    gold_output = F.sigmoid(test_x*scale)#no quant
    print(gold_output)
    # 使用函数方法得到输出
    output_from_function = activation(test_x, scale=scale, type='sigmoid') * scale
    print(output_from_function)
    # 使用类方法得到输出
    pwl_sigmoid = hardpwl('sigmoid')
    pwl_sigmoid.scale = scale.detach()
    output_from_class = pwl_sigmoid(test_x)*scale
    print(output_from_class)
    # 检查输出是否近似相等
    are_outputs_equal = torch.allclose(output_from_function, output_from_class, atol=1e-7)
    print("Are outputs equal?", are_outputs_equal)





