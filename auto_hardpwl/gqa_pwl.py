import torch
import torch.nn as nn
import json

class QuanPWL(nn.Module):
    def __init__(self, pwl_type, pwl_dir) -> None:
        super(QuanPWL, self).__init__()
        # self.hswish = nn.Hardswish()
        self.pwl_hswish = hardpwl_json(pwl_type=pwl_type, pwl_dir=pwl_dir)
    def forward(self, input, scale_x, fix=None):
        self.pwl_hswish.scale = scale_x.detach()
        act = self.pwl_hswish(input, fix=fix)
        return act
        
class hardpwl_json(nn.Module):
    def __init__(self, pwl_type:str, pwl_dir:str) -> None:
        super(hardpwl_json, self).__init__()
        # fp func
        self.act_funcs = {
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'hswish': nn.Hardswish(),
            'exp': torch.exp
        }
        with open(pwl_dir, 'r') as f:
            params = json.load(f)
        self.pwl_funcs = params[pwl_type]
        self.pwl_type = pwl_type
        self.scale = torch.tensor(0.25, requires_grad=False)
        self.slopes = None
        self.intercepts = None
        self.change_pts = None
        self.slopes_scale = None
        self.intercepts_scale = None
     
    def forward(self, input, fix=None):
        device = input.device
        scale = self.scale.to(device)
        # fp func
        func = self.act_funcs[self.pwl_type]
        # pwl func
        # 16 is the bitwidth of final seg_point
        # 3 is the bitwidth of sign and integer part
        scale_bit = -torch.log2(scale).int().item() 
        if scale_bit == -1: scale_bit = 0
        elif scale_bit >5: scale_bit = 5
        decimal_bit = scale_bit 
        if fix is not None: scale_bit = fix
        # print(decimal_bit)
        if decimal_bit > 13:
            raise ValueError(f"decimal_bit is {decimal_bit}, which is greater than 13. Error in layer: {self.__class__.__name__}")
        params = self.pwl_funcs[f'{scale_bit}']
        seg_point_scale = torch.tensor(params['split_points']).to(device)
        intercept_scale = round_to_nearest_bits_torch(torch.tensor(params['bias']).to(device), 6) / scale
        coeff_scale = round_to_nearest_bits_torch(torch.tensor(params['coeff']).to(device), 6)
        # npz_logging
        self.slopes_scale = torch.tensor([1/64.0], dtype=torch.float32).to(device)
        self.intercepts_scale = torch.tensor([1/64.0], dtype=torch.float32).to(device)
        self.slopes = coeff_scale / self.slopes_scale
        self.intercepts = intercept_scale / self.intercepts_scale
        self.change_pts = seg_point_scale
        # 设定指定的位数上限
        if self.pwl_type == 'exp':
            max_binary_digits = 3  # 你可以根据需要修改这个值
        else:
            max_binary_digits = 2  # 你可以根据需要修改这个值
        # 将`seg_point_scale`中二进制整数部分位数超过指定值的元素删除
        for i in range(seg_point_scale.shape[0]):
            if i >= seg_point_scale.shape[0]: break
            sp_decimal_bit = torch.log2(torch.floor(seg_point_scale[i].abs()))
            if sp_decimal_bit > max_binary_digits:
                print("eliminate index:", i)
                if i == seg_point_scale.shape[0] - 1:
                    seg_point_scale = seg_point_scale[:i]
                    intercept_scale = intercept_scale[:i + 1]
                    coeff_scale = coeff_scale[:i + 1]
                else:
                    seg_point_scale = torch.cat([seg_point_scale[:i], seg_point_scale[i + 1:]], dim=-1)
                    intercept_scale = torch.cat([intercept_scale[:i], intercept_scale[i + 1:]], dim=-1)
                    coeff_scale = torch.cat([coeff_scale[:i], coeff_scale[i + 1:]], dim=-1)
        seg_point_scale = round_to_nearest_bits_torch(seg_point_scale, decimal_bit)
        seg_point_scale = seg_point_scale / scale
        pwl_func = torch.zeros_like(input).to(device)
        mask = input.lt(seg_point_scale[0])
        pwl_func = torch.where(mask, intercept_scale[0] + coeff_scale[0] * input, pwl_func)
        for i in range(1, len(seg_point_scale)):
            mask = input.ge(seg_point_scale[i-1]) & input.lt(seg_point_scale[i])
            pwl_func = torch.where(mask, intercept_scale[i] + coeff_scale[i] * input, pwl_func)
        mask = input.ge(seg_point_scale[-1])
        pwl_func = torch.where(mask, intercept_scale[-1] + coeff_scale[-1] * input, pwl_func)
        return (pwl_func * scale - func(input *scale)).detach() + func(input * scale)

def round_to_nearest_bits_torch(x, decimal_bits):
    scaled_value = x * (2 ** decimal_bits)
    rounded_value = torch.round(scaled_value) # very important
    result = rounded_value / (2 ** decimal_bits)
    return result