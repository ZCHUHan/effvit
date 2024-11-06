from calendar import c
import numpy as np
from typing import Dict, List

import torch
import torch.nn as nn

from models.utils import build_kwargs_from_config
from models.nn import DSConv, MBConv, OpSequential
from models.nn import QConvLayer, QEfficientViTBlock, QDSConv, QMBConv
from models.nn.quant_lsq import QuanConv, get_next_global_idx, get_global_idx
from models.nn.lsq import LsqQuantizer4input
__all__ = [
    "Quan_EfficientViTBackbone",
    "quan_efficientvit_backbone_b0",
    "quan_efficientvit_backbone_b1",
    "quan_efficientvit_backbone_b2",
    "quan_efficientvit_backbone_b3",
]
nbit_w = 8
nbit_a = 8
quan_a = 'lsq'
quan_w = 'lsq'
def set_fix(m, setfix):
    children = list(m.named_children())
    idx = 0
    while idx < len (children):
        name, child = children[idx]
        # print(name)
        if isinstance(child, QuanConv):
            child.set_fix(setfix)
        else:
            set_fix(child, setfix)
        idx = idx + 1
        




per_channel = True
class Quan_EfficientViTBackbone(nn.Module):
    def __init__(self, width_list: List[int], depth_list: List[int], in_channels=3, dim=32, branch=False, expand_ratio=4, split_height=4, norm="bn2d", act_func="quanhswish") -> None:
        super().__init__()
        self.split_height = split_height
        self.width_list = []
        self.branch = branch
        # input stem
        # 3x3 convolution with stride 2
        self.input_stem = [
            QConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
                per_channel=per_channel,
                nbit_a=nbit_a,
                nbit_w=nbit_w,
                quan_a=quan_a,
                quan_w=quan_w
            )
        ]
        for _ in range(depth_list[0]):
            block = self.Qbuild_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                per_channel=per_channel,
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
                nbit_a=nbit_a,
                nbit_w=nbit_w,
                quan_a=quan_a,
                quan_w=quan_w,
                res = True
            )
            # self.input_stem.append(QResidualBlock(block, IdentityLayer()))
            self.input_stem.append(block)
        in_channels = width_list[0]
        # self.input_stem = OpSequential(self.input_stem)
        self.input_stem = nn.ModuleList(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.Qbuild_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    per_channel=per_channel,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                    nbit_a=nbit_a,
                    nbit_w=nbit_w,
                    quan_a=quan_a,
                    quan_w=quan_w,
                    res = True if stride == 1 else False
                )
                # block = QResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.Qbuild_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                per_channel=per_channel,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
                nbit_a=nbit_a,
                nbit_w=nbit_w,
                quan_a=quan_a,
                quan_w=quan_w,
                psum_quan=False,
                cg=32
            )
            stage.append(block)
            in_channels = w

            for _ in range(d):
                stage.append(
                    QEfficientViTBlock(
                        in_channels=in_channels,
                        per_channel=per_channel,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                        nbit_a=nbit_a,
                        nbit_w=nbit_w,
                        quan_a=quan_a,
                        quan_w=quan_w,
                        psum_quan=False,
                        cg=32
                    )
                )
            # self.stages.append(OpSequential(stage))
            self.stages.append(nn.ModuleList(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(in_channels: int, out_channels: int, stride: int, expand_ratio: float, norm: str, act_func: str, fewer_norm: bool = False) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:      
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    
    @staticmethod
    def Qbuild_local_block(in_channels: int, out_channels: int, stride: int, expand_ratio: float,
                           norm: str, act_func: str, fewer_norm: bool = False, nbit_w=16, nbit_a=16,
                           quan_a='acy', quan_w='lsq', psum_quan=False, cg=32, per_channel=True, res=False) -> nn.Module:
        if expand_ratio == 1:
            block = QDSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                per_channel=per_channel,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
                nbit_w=nbit_w,
                nbit_a=nbit_a,
                quan_a=quan_a,
                quan_w=quan_w,
                psum_quan=psum_quan,
                cg=cg,
                res=res
            )
        else:      
            block = QMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                per_channel=per_channel,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
                nbit_w=nbit_w,
                nbit_a=nbit_a,
                quan_a=quan_a,
                quan_w=quan_w,
                psum_quan=psum_quan,
                cg=cg,
                res=res
            )
        return block
    
    def set_net_fix(self, setfix):
        set_fix(self, setfix)
     
    def split_and_concat(self, x, split_height, ops, n=4):
        tensor_parts = [x[:,:,i*split_height:(i+1)*split_height,:] for i in range(n)]
        conv_parts = [ops(tensor_parts[i]) for i in range(len(tensor_parts))]
        concat_tensor = torch.cat(conv_parts, dim=2)
        return concat_tensor
    
    def forward(self, x: torch.Tensor, fix_bn=False) -> Dict[str, torch.Tensor]:
        if not self.branch:
            output_dict = {"input": x}
            # print("input shape: ", x.shape)
            x_stem = self.input_stem[0](x)
            for op in self.input_stem[1:]:
                x_stem = op(x_stem)
            output_dict["stage0"] = concat_tensor = x_stem
            for stage_id, stage in enumerate(self.stages, 1):
                if stage_id in (1,2):
                    concat_tensor = stage(concat_tensor)
                    output_dict["stage%d" % stage_id] = concat_tensor 
                elif stage_id == 3:
                    for ops in stage:
                        # concat_tensor = self.split_and_concat(x=concat_tensor, split_height=concat_tensor.shape[2] // 2, ops=ops, n=self.split_height)
                        concat_tensor = ops(concat_tensor)
                    output_dict["stage%d" % stage_id] = concat_tensor
                else:
                    for ops in stage:
                        # concat_tensor = self.split_and_concat(x=concat_tensor, split_height=concat_tensor.shape[2] // 2, ops=ops, n=self.split_height)
                        concat_tensor = ops(concat_tensor)
                    output_dict["stage%d" % stage_id] = concat_tensor 
            output_dict["stage_final"] = concat_tensor
            return output_dict
        else:
            output_dict = {"input": x}
            x_stem = self.input_stem[0](x)
            # for op in self.input_stem[1:]:
            #     x_stem = op(x_stem)
            x_stem_0 = x_stem[:,:,0:x_stem.shape[2]//4,:]
            x_stem_1 = x_stem[:,:,x_stem.shape[2]//4: 2*x_stem.shape[2]//4,:]
            x_stem_2 = x_stem[:,:,2*x_stem.shape[2]//4: 3*x_stem.shape[2]//4,:]
            x_stem_3 = x_stem[:,:,3*x_stem.shape[2]//4: 4*x_stem.shape[2]//4,:]
            for op in self.input_stem[1:]:
                x_stem_0 = op(x_stem_0)
                x_stem_1 = op(x_stem_1)
                x_stem_2 = op(x_stem_2)
                x_stem_3 = op(x_stem_3)
            
            for stage_id, stage in enumerate(self.stages, 1):
                if stage_id == 1:
                    # block_0 = stage.op_list[0]
                    # block_1 = stage.op_list[1]
                    # tensor_parts = [x_stem_0, x_stem_1, x_stem_2, x_stem_3]
                    # concat_tensor = [block_0(tensor_parts[i]) for i in range(len(tensor_parts))]
                    # concat_tensor = torch.cat(concat_tensor, dim=2)
                    # if not self.training and get_global_idx() >= 0: #log npz:
                    #     # record idx of concat npz file
                    #     idx_concat_1 = get_next_global_idx()
                    #     print("idx_concat: ", idx_concat_1)
                    # concat_tensor = block_1(concat_tensor)
                    # # add scale_qkv value to idx_concat concat npz file
                    # if not self.training and get_global_idx() >= 0:
                    #     np.savez("npz_logging/" + str(idx_concat_1) + "_concat", scale_qkv=block_1.inverted_conv.conv.scale_a.cpu().numpy())
                    # output_dict["stage%d" % stage_id] = concat_tensor
                    tensor_parts = [x_stem_0, x_stem_1, x_stem_2, x_stem_3]
                    concat_tensor = [stage(tensor_parts[i]) for i in range(len(tensor_parts))]
                    output_dict["stage%d" % stage_id] = concat_tensor
                elif stage_id ==2:
                    # concat_tensor = self.split_and_concat(x=concat_tensor, split_height=concat_tensor.shape[2] // self.split_height, ops=stage[0], n=self.split_height)
                    # for ops in stage[1:]:
                    block_0 = stage.op_list[0]
                    block_1 = stage.op_list[1]
                    concat_tensor = [block_0(concat_tensor[i]) for i in range(len(concat_tensor))]
                    concat_tensor = torch.cat(concat_tensor, dim=2)
                    if not self.training and get_global_idx() >= 0: #log npz:
                        # record idx of concat npz file
                        idx_concat_1 = get_next_global_idx()
                        print("idx_concat: ", idx_concat_1)
                    concat_tensor = block_1(concat_tensor)
                    if not self.training and get_global_idx() >= 0:
                        np.savez("npz_logging/" + str(idx_concat_1) + "_concat", scale_qkv=block_1.inverted_conv.conv.scale_a.cpu().numpy())
                    output_dict["stage%d" % stage_id] = concat_tensor
                else:
                    for ops in stage:
                        # concat_tensor = self.split_and_concat(x=concat_tensor, split_height=concat_tensor.shape[2] // 2, ops=ops, n=self.split_height)
                        concat_tensor = ops(concat_tensor)
                    output_dict["stage%d" % stage_id] = concat_tensor 
            output_dict["stage_final"] = concat_tensor
            return output_dict
        # output_dict = {"input": x}
        # output_dict["stage0"] = x = self.input_stem(x)
        # for stage_id, stage in enumerate(self.stages, 1):
        #     output_dict["stage%d" % stage_id] = x = stage(x)
        # output_dict["stage_final"] = x
        # return output_dict


def quan_efficientvit_backbone_b0(**kwargs) -> Quan_EfficientViTBackbone:
    backbone = Quan_EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, Quan_EfficientViTBackbone),
    )
    return backbone

def quan_efficientvit_backbone_b0_demo(**kwargs) -> Quan_EfficientViTBackbone:
    backbone = Quan_EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        branch=True,
        **build_kwargs_from_config(kwargs, Quan_EfficientViTBackbone),
    )
    return backbone

def quan_efficientvit_backbone_b1(**kwargs) -> Quan_EfficientViTBackbone:
    backbone = Quan_EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, Quan_EfficientViTBackbone),
    )
    return backbone


def quan_efficientvit_backbone_b2(**kwargs) -> Quan_EfficientViTBackbone:
    backbone = Quan_EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, Quan_EfficientViTBackbone),
    )
    return backbone


def quan_efficientvit_backbone_b3(**kwargs) -> Quan_EfficientViTBackbone:
    backbone = Quan_EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, Quan_EfficientViTBackbone),
    )
    return backbone
