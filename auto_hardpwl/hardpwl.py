from importlib.metadata import requires
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import json
class hardpwl_json(nn.Module):
    def __init__(self, pwl_type:str, pwl_dir:str) -> None:
        super(hardpwl_json, self).__init__()
        # fp func
        self.act_funcs = {
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'hswish': nn.Hardswish()
        }
        with open(pwl_dir, 'r') as f:
            params = json.load(f)
        self.pwl_funcs = params[pwl_type]
        self.pwl_type = pwl_type
        self.scale = torch.tensor(0.25, requires_grad=False)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        scale = self.scale.to(device)
        # fp func
        func = self.act_funcs[self.pwl_type]
        # pwl func
        # 16 is the bitwidth of final seg_point
        # 3 is the bitwidth of sign and integer part
        scale_bit = torch.abs(torch.log2(scale)).int().item() 
        decimal_bit = 16 - 3 - scale_bit 
        if decimal_bit > 13:
            raise ValueError(f"decimal_bit is {decimal_bit}, which is greater than 13. Error in layer: {self.__class__.__name__}")
        params = self.pwl_funcs[f'{decimal_bit}']
        seg_point_scale = torch.tensor(params['split_points']).to(device) / scale
        intercept_scale = torch.tensor(params['bias']).to(device) / scale
        coeff_scale = torch.tensor(params['coeff']).to(device)
        pwl_func = torch.zeros_like(input).to(device)
        mask = input.lt(seg_point_scale[0])
        pwl_func = torch.where(mask, intercept_scale[0] + coeff_scale[0] * input, pwl_func)
        for i in range(len(seg_point_scale)):
            mask = input.ge(seg_point_scale[i-1]) & input.lt(seg_point_scale[i])
            pwl_func = torch.where(mask, intercept_scale[i] + coeff_scale[i] * input, pwl_func)
        mask = input.ge(seg_point_scale[-1])
        pwl_func = torch.where(mask, intercept_scale[-1] + coeff_scale[-1] * input, pwl_func)
        return (pwl_func * scale - func(input *scale)).detach() + func(input * scale)


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
            'hswish':{
                'coeff': torch.tensor([-0.0,
                                       -0.359375,
                                       -0.0625,
                                       0.28125,
                                       0.625,
                                       0.953125,
                                       1.296875,
                                       1.0], dtype=torch.float32),
                'bias': torch.tensor([0.0,
                                          -1.09375,
                                          -0.453125,
                                          -0.03125,
                                          0.015625,
                                          -0.28125,
                                          -0.921875,
                                          -0.0], dtype=torch.float32),
                'split_points': torch.tensor([-3.0,
                                           -2.1787109375,
                                           -1.228515625,
                                           -0.130859375,
                                           0.89453125,
                                           1.83740234375,
                                           2.98828125], dtype=torch.float32)
            }
        }
        self.pwl_type = pwl_type
        self.scale = torch.tensor(0.25, requires_grad=False)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        # Get the correct activation parameters and function
        params = self.pwl_funcs[self.pwl_type]
        func = self.act_funcs[self.pwl_type]
        scale = self.scale.to(device)
        seg_point_scale = params['split_points'].to(device) / scale
        intercept_scale = params['bias'].to(device) / scale
        # pwl func
        pwl_func = torch.zeros_like(input).to(device)
        mask = input.lt(seg_point_scale[0])
        pwl_func = torch.where(mask, intercept_scale[0] + params['coeff'][0] * input, pwl_func)
        for i in range(len(seg_point_scale)):
            mask = input.ge(seg_point_scale[i-1]) & input.lt(seg_point_scale[i])
            pwl_func = torch.where(mask, intercept_scale[i] + params['coeff'][i] * input, pwl_func)
        mask = input.ge(seg_point_scale[-1])
        pwl_func = torch.where(mask, intercept_scale[-1] + params['coeff'][-1] * input, pwl_func)
        return (pwl_func * scale - func(input *scale)).detach() + func(input * scale)
 

if __name__ == '__main__': 
    hardpwl_json_model = hardpwl_json(pwl_type='hswish', pwl_dir='./params_pwl/hswish_pwl.json')
    hardpwl_model = hardpwl(pwl_type='hswish')
    input_tensor = torch.tensor([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]])
    output1 = hardpwl_json_model(input_tensor)
    output2 = hardpwl_model(input_tensor)
    print(output1)
    print(output2)
    is_close = torch.allclose(output1, output2, atol=1e-6)
    print("Outputs are close:", is_close)
