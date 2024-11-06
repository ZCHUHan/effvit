
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_same_padding, resize, val2list, val2tuple, merge_tensor
from models.nn.norm import build_norm
from models.nn.act import build_act, Quanhswish
from models.nn.resize_new_pytorch import resize_new
from models.nn.quant_lsq import QuanConv, PActFn, PACT, SymmetricQuantFunction, get_next_global_idx, get_global_idx
from models.nn.lsq import LsqQuantizer4input, LsqQuantizer4weight
from models.nn.ops import ConvLayer, DSConv, MBConv, EfficientViTBlock, OpSequential, ResidualBlock, IdentityLayer, LiteMSA
__all__ = [
    "QConvLayer",
    # "UpSampleLayer",
    # "DownSampleLayer",
    # "LinearLayer",
    # "IdentityLayer",
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


class QConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=None,
        groups=1,
        use_bias=False,
        dropout_rate=0,
        norm="bn2d",
        act_func="relu",
        per_channel=True,
        quan_a='acy',
        quan_w='lsq',
        nbit_w=8,
        nbit_a=8,
        res = False,
        psum_quan=False,
        cg=32
    ):
        super(QConvLayer, self).__init__()

        padding = padding if padding else get_same_padding(kernel_size)
        padding *= dilation
        self.res = res
        self.dropout = nn.Dropout2d(dropout_rate, inplace=False) if dropout_rate > 0 else None
        self.quan_a_name = quan_a
        if self.quan_a_name == 'acy':
            self.pact = PACT(nbit_a)
        self.conv = QuanConv(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            quan_name_w=quan_w,
            quan_name_a=self.quan_a_name,
            nbit_w=nbit_w,
            nbit_a=nbit_a,
            per_channel=per_channel,
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
            norm=norm,
            psum_quan=psum_quan,
            cg=cg,
        )
       
        # self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)

        if self.quan_a_name == 'acy' and self.res == False:
            x, scale_a = self.pact(x)
            self.conv.scale_a = scale_a.detach()
        elif self.quan_a_name == 'lsq' and self.res == False:
            x, scale_a = self.conv.lsq_a(x)
            self.conv.scale_a = scale_a.detach()

        x = self.conv(x)

        if self.act:
            x = self.act(x)
        
        return x

            


class QUpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: Union[int, Tuple[int, int], List[int], None] = None,
        factor=2,
        align_corners=False,
        nbit_a=8,
        all_positive=False,
        quan_a_name = 'lsq'
    ):
        super(QUpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners
        self.quan_bf = LsqQuantizer4input(
                        nbit=nbit_a,
                        all_positive=False,
                        per_channel=False
                    ) if quan_a_name == 'lsq' else PACT(nbit_a)

        # self.quan_af = LsqQuantizer4input(
        #                 nbit=nbit_a,
        #                 all_positive=False,
        #                 per_channel=False
        #             ) if quan_a_name == 'lsq' else PACT(nbit_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, scale_bf = self.quan_bf(x)
        x_r = resize(x, self.size, self.factor, self.mode, self.align_corners)
        x_r, scale_r = self.quan_bf(x_r)
        assert(scale_bf == scale_r)
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx) + "_resize", input=x.cpu().numpy(), input_scale=scale_bf.cpu().numpy(), output= x_r.cpu().numpy())
        # x_af = x_r
        # x_af, scale_af = self.quan_af(x_r) 
        return x_r


class DownSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: Union[int, Tuple[int, int], List[int], None] = None,
        factor=0.5,
        align_corners=False,
    ):
        super(DownSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(x, self.size, self.factor, self.mode, self.align_corners)
    
class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout_rate=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)
    
    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class QDSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
        per_channel=True,
        nbit_w=8,
        nbit_a=8,
        psum_quan=False,
        cg=32,
        res = False,
        quan_a='acy',
        quan_w='lsq'
    ):
        super(QDSConv, self).__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a  
        self.res = res
        self.lsq_a = LsqQuantizer4input(bit=self.nbit_a, all_positive=False, per_channel=False) if quan_a == 'lsq' else PACT(nbit_a) 
        self.lsq_out = LsqQuantizer4input(bit=16, all_positive=False, per_channel=False) if quan_a == 'lsq' else PACT(nbit_a) 
        self.depth_conv = QConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
            per_channel=per_channel,
            nbit_w=nbit_w,
            nbit_a=nbit_a,
            quan_a=quan_a,
            quan_w=quan_w,
            res=res
        )
        self.point_conv = QConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
            per_channel=per_channel,
            nbit_w=nbit_w,
            nbit_a=nbit_a,
            quan_a=quan_a,
            quan_w=quan_w,
            psum_quan=psum_quan,
            cg=cg,
            res=False
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            x_res, scale_res  = self.lsq_a(x)
            self.depth_conv.conv.scale_a = scale_res.detach()
            x_o = self.depth_conv(x_res)
        else:
            x_o = self.depth_conv(x)
        x_o = self.point_conv(x_o)
        # if self.res:
        #     x_o, _ = self.lsq_a(x_o)
        #     x_o = x_res + x_o
        if self.res:
            x_1, x_o_scale = self.lsq_out(x_o)
            x_res, x_res_scale = self.lsq_out(x_res)
            x_o = x_res + x_1
            if not self.training and get_global_idx() >= 0: #log npz:
                idx = get_next_global_idx()
                np.savez("npz_logging/" + str(idx) + "_add", input1=x_1.cpu().numpy(), input1_scale=x_o_scale.cpu().numpy(), input2=x_res.cpu().numpy(), output=x_o.cpu().numpy(), input2_scale=x_res_scale.cpu().numpy())
        return x_o

class QMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int, # 32
        out_channels: int, # 32 
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6, # 4
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
        per_channel=True,
        nbit_w=8,
        nbit_a=8,
        psum_quan=False,
        cg=32,
        res = False,
        quan_a='acy',
        quan_w='lsq'
    ):
        super(QMBConv, self).__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a   
        self.res = res
        self.lsq_a = LsqQuantizer4input(bit=self.nbit_a, all_positive=False, per_channel=False) if quan_a == 'lsq' else PACT(nbit_a) 
        self.lsq_out = LsqQuantizer4input(bit=16, all_positive=False, per_channel=False) if quan_a == 'lsq' else PACT(nbit_a) 
        self.inverted_conv = QConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
            per_channel=per_channel,
            nbit_w=nbit_w,
            nbit_a=nbit_a,
            quan_a=quan_a,
            quan_w=quan_w,
            res=res
        )
        self.depth_conv = QConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
            per_channel=per_channel,
            nbit_w=nbit_w,
            nbit_a=nbit_a,
            quan_a=quan_a,
            quan_w=quan_w,
            res=False
        )
        self.point_conv = QConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
            per_channel=per_channel,
            nbit_w=nbit_w,
            nbit_a=nbit_a,
            quan_a=quan_a,
            quan_w=quan_w,
            psum_quan=psum_quan,
            cg=cg,
            res=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            x_res, scale_res  = self.lsq_a(x)
            self.inverted_conv.conv.scale_a = scale_res.detach()
            x_o = self.inverted_conv(x_res)
        else:
            x_o = self.inverted_conv(x)
        x_o = self.depth_conv(x_o)
        x_o = self.point_conv(x_o)
        # if self.res:
        #     x_o, _ = self.lsq_out(x_o)
        #     x_o = x_res + x_o
        if self.res:
            x_1, x_o_scale = self.lsq_out(x_o)
            x_res, x_res_scale = self.lsq_out(x_res)
            x_o = x_res + x_1
            if not self.training and get_global_idx() >= 0: #log npz:
                idx = get_next_global_idx()
                np.savez("npz_logging/" + str(idx) + "_add", input1=x_1.cpu().numpy(), input1_scale=x_o_scale.cpu().numpy(), input2=x_res.cpu().numpy(), output=x_o.cpu().numpy(), input2_scale=x_res_scale.cpu().numpy())
        return x_o


class QLiteMSA(nn.Module):
    r""" Quantized Lightweight Multi-Scale Attention """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: Tuple[int, ...] = (5,),
        per_channel=True,
        nbit_w=8,
        nbit_a=8,
        quan_a='acy',
        quan_w='lsq'
    ):
        super(QLiteMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)
        self.per_channel = per_channel
        total_dim = heads * dim
        self.nbit_a = nbit_a
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.dim = dim
        self.quan_a_name = quan_a 
        if self.quan_a_name == 'acy':
            self.quan_a = PACT(nbit_a)
            self.pact = self.quan_a
        elif self.quan_a_name == 'lsq':
            self.quan_a = LsqQuantizer4input(
                            bit=nbit_a,
                            all_positive=False,
                            per_channel=False
                        )
            self.lsq_out = LsqQuantizer4input(bit=16, all_positive=False, per_channel=False) if quan_a == 'lsq' else PACT(nbit_a) 
            # self.lsq = self.quan_a
        self.qkv_layers = nn.ModuleDict({
            name: 
                QuanConv(
                    in_channels,
                    total_dim,
                    kernel_size=1,
                    quan_name_w=quan_w,
                    quan_name_a=self.quan_a_name,
                    nbit_w=nbit_w,
                    nbit_a=nbit_a,
                    per_channel=per_channel,
                    norm=norm[0],
                    bias=use_bias[0],
                )
            for name in ["q", "k", "v"]
        })
        self.act = build_act(act_func[0]) 
        self.aggreg = nn.ModuleDict({
            name: nn.ModuleList(
                [
                    nn.Sequential(
                        QConvLayer(
                            total_dim,
                            total_dim,
                            scale, 
                            padding=get_same_padding(scale), 
                            groups=total_dim,
                            use_bias=use_bias[0],
                            per_channel=per_channel,
                            nbit_w=nbit_w,
                            nbit_a=nbit_a,
                            quan_w=quan_w,
                            quan_a=quan_a,
                            norm=None,
                            act_func=None
                        ),
                        QConvLayer(
                            total_dim,
                            total_dim,
                            1, 
                            groups=heads,
                            use_bias=use_bias[0], 
                            per_channel=per_channel,
                            nbit_w=nbit_w,
                            nbit_a=nbit_a,
                            quan_w=quan_w,
                            quan_a=quan_a,
                            norm=None,
                            act_func=None
                        )
                    )
                    for scale in scales
                ]
            )
            for name in ["q", "k", "v"]
        })

        self.concat = nn.ModuleDict({
            name: LsqQuantizer4input(
                bit=nbit_a,
                per_channel=False,
                all_positive=False 
            ) if self.quan_a_name == 'lsq' else PACT(nbit_a)
            for name in ["q", "k", "v"]
        })

        self.scale_concat = [0.0, 0.0 , 0.0]

        self.kv_a    = LsqQuantizer4input(bit=nbit_a, per_channel=False, all_positive=False) if self.quan_a_name == 'lsq' else PACT(nbit_a)
        self.qkv_a   = LsqQuantizer4input(bit=nbit_a, per_channel=False, all_positive=False) if self.quan_a_name == 'lsq' else PACT(nbit_a)
        self.quan_sum   = LsqQuantizer4input(bit=8, update=False, learnable=False, per_channel=False, all_positive=False) if self.quan_a_name == 'lsq' else PACT(nbit_a)
        self.quan_denorm   = LsqQuantizer4input(bit=16, update=False, per_channel=False, all_positive=False) if self.quan_a_name == 'lsq' else PACT(nbit_a)
        self.kernel_func = build_act(kernel_func, inplace=False)
        self.proj = QConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            per_channel=per_channel,
            nbit_w=nbit_w,
            nbit_a=nbit_a,
            quan_w=quan_w,
            quan_a=quan_a
        )
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())
        # quantize input
        x_res, scale_a = self.quan_a(x)
        self.qkv_layers["q"].scale_a = scale_a.detach()
        self.qkv_layers["k"].scale_a = scale_a.detach()
        self.qkv_layers["v"].scale_a = scale_a.detach()
        # quantize qkv
        q = self.qkv_layers["q"](x_res)
        k = self.qkv_layers["k"](x_res)
        v = self.qkv_layers["v"](x_res)
        # scale-1 qkv
        qkv_dict = {
            "q": self.act(q) if self.act is not None else q,
            "k": self.act(k) if self.act is not None else k,
            "v": self.act(v) if self.act is not None else v
        }
        concat_qkv = {}
        scale_qkv  = {}
        for name in ["q", "k", "v"]:
            # obtain scale-1 qkv
            x_qkv = qkv_dict[name]
            # ensure concat's input scales are the same
            self.aggreg[name][0][0].conv.lsq_a = self.concat[name]
            # scale-2 qkv
            x1 = self.aggreg[name][0][0](x_qkv)
            x2 = self.aggreg[name][0][1](x1)
            # concat scale-1&2 qkv
            concat_x = torch.cat((x_qkv, x2), dim=1)
            # quantize for sharing same scale
            concat_qkv[name], scale_qkv[name] = self.concat[name](concat_x)
            # reshape to B, heads, H*W, dim(8)
            concat_qkv[name] = concat_qkv[name].reshape(B, -1, self.dim, H*W).contiguous().transpose(-1,-2).contiguous()
            # only relu on q and k
            concat_qkv[name] = self.kernel_func(concat_qkv[name] / scale_qkv[name]) * scale_qkv[name] if name != "v" else concat_qkv[name] 
            
            if not self.training and get_global_idx() >= 0: #log npz:
                idx = get_next_global_idx()
                np.savez("npz_logging/" + str(idx) + "_concat", concat_qkv=concat_qkv[name].cpu().numpy(), scale_qkv=scale_qkv[name].cpu().numpy())
                
        kv = torch.matmul(concat_qkv["k"].transpose(-1, -2), concat_qkv["v"])
        # quantize relu(k^T) * v
        kv1, kv_scale1 = self.kv_a(kv)

        qkv = torch.matmul(concat_qkv["q"], kv1) 
        # quantize qkv
        qkv, qkv_scale = self.qkv_a(qkv)

        k_sum = torch.sum(concat_qkv["k"].transpose(-1, -2), dim=-1, keepdim=True)
        if not self.training:
            k_sum, k_sum_s = self.quan_sum(k_sum)
        denom = torch.matmul(concat_qkv["q"], k_sum) 
        if not self.training:
            denom, denom_s = self.quan_denorm(denom)
        
        # if get_global_idx() >= 0: #log npz:
        #     # out = qkv / denom
        #     out = qkv / (denom + 1e-15)

        # else:
        #     out = qkv / (denom + 1e-15)
        
        scale = 512
        if not self.training and get_global_idx() >= 0: #log npz:
            divider = ((denom) / denom_s)
            # set divider to 1 if it is too small
            divider = torch.where(divider < 1e-15, torch.ones_like(divider), divider)
            out = torch.trunc(torch.clip((qkv / qkv_scale) * scale, min=-32768, max=32767) / (divider)) * (qkv_scale / denom_s / scale)
        else:
            out = qkv / (denom + 1e-15)

        
        
        # reshape to B, heads, dim(8), H*W
        out = torch.transpose(out, -1, -2)
        # reshape to B, heads*dim(8), H, W
        out = torch.reshape(out, (B, -1, H, W))
        
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx) + "_mul", input_A_scale=scale_qkv["k"].cpu().numpy(), input_A=concat_qkv["k"].transpose(-1, -2).cpu().numpy(), input_B_scale=scale_qkv["v"].cpu().numpy(), input_B=concat_qkv["v"].cpu().numpy(), output=kv1.cpu().numpy(), output_scale=kv_scale1.cpu().numpy())
        
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx) + "_mul", input_A_scale=scale_qkv["q"].cpu().numpy(), input_B_scale=k_sum_s.cpu().numpy(), input_A=concat_qkv["q"].cpu().numpy(), input_B=k_sum.cpu().numpy(), output=denom.cpu().numpy(), output_scale=denom_s.cpu().numpy())
            
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx) + "_mul", input_A_scale=scale_qkv["q"].cpu().numpy(), input_A=concat_qkv["q"].cpu().numpy(), input_B_scale=kv_scale1.cpu().numpy(), input_B=kv1.cpu().numpy(), output=qkv.cpu().numpy(), output_scale=qkv_scale.cpu().numpy())
            
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx) + "_div", out=out.cpu().numpy(), input_A=qkv.cpu().numpy(), input_A_scale=qkv_scale.cpu().numpy(), input_B=denom.cpu().numpy(), input_B_scale=denom_s.cpu().numpy())
            
        # final projection 
        out = self.proj(out)
        out1, out_scale = self.lsq_out(out)
        x_res, x_res_scale = self.lsq_out(x_res)
        # residual
        out = out1 + x_res
        
        if not self.training and get_global_idx() >= 0: #log npz:
            idx = get_next_global_idx()
            np.savez("npz_logging/" + str(idx) + "_add", input1=out1.cpu().numpy(), input1_scale=out_scale.cpu().numpy(), input2=x_res.cpu().numpy(), input2_scale=x_res_scale.cpu().numpy())
        return out


class QEfficientViTBlock(nn.Module):
    def __init__(self, in_channels: int, heads_ratio: float = 1.0, dim=32,
                 expand_ratio: float = 4, norm="bn2d", act_func="hswish",
                 nbit_w=8, nbit_a=8, per_channel=True, quan_a='lsq', quan_w='lsq',
                 psum_quan=False, cg=32):
        super(QEfficientViTBlock, self).__init__()
        self.context_module = QLiteMSA(
            in_channels=in_channels,
            out_channels=in_channels,
            heads_ratio=heads_ratio,
            dim=dim,
            norm=(None, norm),
            per_channel=per_channel,
            nbit_w=nbit_w,
            nbit_a=nbit_a,
            quan_a=quan_a,
            quan_w=quan_w
        )
           
        self.local_module = QMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
            per_channel=per_channel,
            nbit_w=nbit_w,
            nbit_a=nbit_a,
            quan_a=quan_a,
            quan_w=quan_w,
            psum_quan=psum_quan,
            cg=cg,
            res=True
        )
        # self.local_module = QResidualBlock(local_module, IdentityLayer(), nbit_a=nbit_a)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class QResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
        nbit_a=8
    ):
        super(QResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)
        self.res_quan = LsqQuantizer4input(
                            bit=nbit_a,
                            all_positive=False,
                            per_channel=False
                        )
        
    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    # 0705
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            # res = self.forward_main(x) + self.shortcut(x)
            out = self.forward_main(x)
            if self.main.quan_a_name == 'acy':
                pact = self.main.pact
                out, _ = pact(out)
                res, _ = pact(self.shortcut(x))
                res = out + res
                # res = pact(out) + self.shortcut(pact(x))
            # elif self.main.quan_a_name == 'lsq':
            #     lsq = self.main.lsq
            #     out, _ = lsq(out)
            #     # out, _ = self.quan_a(out)
            #     res, _ = lsq(self.shortcut(x))
            #     res = out + res
            elif self.main.quan_a_name == 'lsq':
                lsq = self.main.lsq
                out, _ = self.res_quan(out)
                print('dpc')
                # out, _ = self.quan_a(out)
                res, _ = lsq(self.shortcut(x))
                res = out + res
            if self.post_act:
                res = self.post_act(res)
        return res
    

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
        a = {}
        a['ss'] = ConvLayer(1, 1, 1)
        print(type(a))
        print(type(a.keys()))
        self.a_keys = list(a.keys())
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


class QOpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super(QOpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
