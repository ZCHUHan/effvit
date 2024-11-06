import re
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# activation definition
ACT_FUNCS = {
    "swish": lambda x: x / (1.0 + np.exp(-x)) if -3.0 < x < 3.0 else (0 if x <= -3.0 else x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)) if -3.0 < x < 3.0 else (0 if x <= -3.0 else 1.0),
    "tanh": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) if -3.0 < x < 3.0 else (-1.0 if x <= -3.0 else 1.0),
    "gelu": lambda x: 0.5 * x * (1 + special.erf(x / np.sqrt(2))) if -3.0 < x < 3.0 else (0 if x <= -3.0 else x),
    "hswish": lambda x: x * np.clip(x + 3, 0, 6) / 6
}

def calculate_coeff_bias(a1, a2, bit, act_type):
    func = ACT_FUNCS[act_type]
    coeff = (func(a2) - func(a1)) / (a2 - a1)
    bias = -a1 * coeff + func(a1)
    resize = 2**(bit-2)
    return (np.round(coeff*resize)/resize, np.round(bias*resize)/resize)

def get_list(split_point, a_bit, act_type): 
    coeff_bias_pairs = [calculate_coeff_bias(a1, a2, a_bit, act_type) for a1, a2 in zip(split_point[:-1], split_point[1:])]
    coeff, bias = zip(*coeff_bias_pairs)
    print(f"slope is :{coeff}, len is {len(coeff)}")
    print(f"intercept is :{bias}, len is {len(bias)}")
    print("seg_point is :", split_point[1:-1])
    print("len of seg_point is :", len(split_point[1:-1]))
    return coeff, bias

def piecewise_linear_approximation(x, split_points, coeff, bias):
    if x < split_points[0]:
        return coeff[0] * x + bias[0]
    for i in range(len(split_points) - 1):
        if split_points[i] <= x < split_points[i + 1]:
            return coeff[i] * x + bias[i]
    return coeff[-1] * x + bias[-1]


def plot_approximation(func_name, split_points, a_bit=8):
    original_func = ACT_FUNCS[func_name]
    coeff, bias = get_list(split_point=split_points, act_type=func_name, a_bit=a_bit)
    
    x_values = np.linspace(-4, 4, 400)
    original_values = [original_func(x) for x in x_values]
    approx_values = [piecewise_linear_approximation(x, split_points, coeff, bias) for x in x_values]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, original_values, label=f'Original {func_name}', color='blue')
    plt.plot(x_values, approx_values, label='Linear Approximation', linestyle='--', color='red')
    
    plt.scatter(split_points[1:-1], [original_func(sp) for sp in split_points[1:-1]], color='green', marker='o', label='Split Points')
    
    plt.legend()
    plt.title(f'{func_name} and its Linear Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    plt.savefig(f'./pwl_plot/{func_name.lower()}_approximation_{a_bit}bit_{len(split_points)-2}points.png', dpi=300)

class hardpwl(nn.Module):
    def __init__(self, pwl_type:str) -> None:
        super(hardpwl, self).__init__()
        # fp func
        self.act_funcs = {
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'hswish': nn.Hardswish()
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
            },
            'hswish':{
                # 15 bins
                # 'slope': torch.tensor([0.0, -0.453125, -0.328125, -0.203125, -0.125, -0.046875, 0.078125, 0.25, 0.421875, 0.578125, 0.75, 0.921875, 1.125, 1.375, 1.0], dtype=torch.float32),
                # 'intercept': torch.tensor([0.0, -1.375, -1.03125, -0.75, -0.578125, -0.4375, -0.25, -0.078125, 0.0, 0.0, -0.078125, -0.25, -0.5625, -1.125, 0.0], dtype=torch.float32),
                # 'seg_point': torch.tensor([-3, -2.75, -2.25, -2, -1.75, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.25, 3], dtype=torch.float32)
                # 14 bins
                # 'slope': torch.tensor([0.0, -0.421875, -0.25, -0.078125, 0.078125, 0.25, 0.421875, 0.578125, 0.75, 0.921875, 1.078125, 1.25, 1.421875, 1.0], dtype=torch.float32),
                # 'intercept': torch.tensor([0.0, -1.25, -0.828125, -0.5, -0.25, -0.078125, 0.0, 0.0, -0.078125, -0.25, -0.5, -0.828125, -1.25, 0.0], dtype=torch.float32),
                # 'seg_point': torch.tensor([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=torch.float32)
                # 7 bins
                'slope': torch.tensor([0.0, -0.359375, -0.0625, 0.21875, 0.515625, 0.890625, 1.28125, 1.0], dtype=torch.float32),
                'intercept': torch.tensor([0.0, -1.0625, -0.453125, -0.09375, 0.03125, -0.15625, -0.859375, 0.0], dtype=torch.float32),
                'seg_point': torch.tensor([-3.0, -2.125, -1.28125, -0.40625, 0.53125, 1.84375, 3.125], dtype=torch.float32)
            }
        }
        self.pwl_type = pwl_type
        self.scale = torch.ones(1, requires_grad=False)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        # Get the correct activation parameters and function
        params = self.pwl_funcs[self.pwl_type]
        func = self.act_funcs[self.pwl_type]
        scale = self.scale.to(device)
        seg_point_scale = params['seg_point'].to(device) / scale
        intercept_scale = params['intercept'].to(device) / scale
        # pwl func
        pwl_func = torch.zeros_like(input).to(device)
        mask = input.lt(seg_point_scale[0])
        pwl_func = torch.where(mask, intercept_scale[0] + params['slope'][0] * input, pwl_func)
        for i in range(len(seg_point_scale)):
            mask = input.ge(seg_point_scale[i-1]) & input.lt(seg_point_scale[i])
            pwl_func = torch.where(mask, intercept_scale[i] + params['slope'][i] * input, pwl_func)
        mask = input.ge(seg_point_scale[-1])
        pwl_func = torch.where(mask, intercept_scale[-1] + params['slope'][-1] * input, pwl_func)
        return (pwl_func * scale - func(input *scale)).detach() + func(input * scale)
 
if __name__ == '__main__':       
    # the -10000 and 10000 are very important!
    # -10000 represents negative infinity
    # 10000 represents positive infinity
    # plot_approximation("hswish", [-10000, -3, -2.75, -2.25, -2, -1.75, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 10000])
    plot_approximation("hswish", [-10000, -3.0, -1.875, -1.03125, -0.28125, 0.5, 1.53125, 2.28125, 3.0, 10000])
    # plot_approximation("hswish", [-10000, -3, -1, -0.5, 0, 0.5, 1, 3, 10000])

