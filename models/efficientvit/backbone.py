from os import close
from typing import Dict, List

from numpy import block
import torch
import torch.nn as nn

from models.utils import build_kwargs_from_config
from models.nn import ConvLayer, DSConv, MBConv, EfficientViTBlock, OpSequential, ResidualBlock, IdentityLayer
__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b0_demo",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "branch_efficientvit_backbone_b0"
]

# branch = True
class EfficientViTBackbone(nn.Module):
    def __init__(self, width_list: List[int], depth_list: List[int], in_channels=3, dim=32, branch=False, expand_ratio=4, split_height=4, norm="bn2d", act_func="hswish") -> None:
        super().__init__()
        self.branch = branch
        self.width_list = []
        self.split_height = split_height
        # input stem
        # 3x3 convolution with stride 2
        self.input_stem = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
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
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
           )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            # self.stages.append(OpSequential(stage))
            self.stages.append(nn.ModuleList(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(in_channels: int, out_channels: int, stride: int, expand_ratio: float, norm: str, act_func: str, fewer_norm: bool = False, close_dw: bool = False) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
                close_dw=close_dw
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

    def split_and_concat(self, x, split_height, ops, n=4):
        tensor_parts = [x[:,:,i*split_height:(i+1)*split_height,:] for i in range(n)]
        test_tensor = tensor_parts
        conv_parts = [ops(tensor_parts[i]) for i in range(len(tensor_parts))]
        concat_tensor = torch.cat(conv_parts, dim=2)
        return concat_tensor

    def forward(self, x: torch.Tensor, fix_bn=True) -> Dict[str, torch.Tensor]:
        # print(x.shape)
        if not self.branch:
            output_dict = {"input": x}
            x_stem = self.input_stem[0](x)
            # x_stem = self.input_stem[1:](x_stem)
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
            
            # output_dict["stage0"] = concat_tensor = x_stem
            for stage_id, stage in enumerate(self.stages, 1):
                if stage_id == 1:
                    # block_0= stage.op_list[0]
                    # block_1= stage.op_list[1]
                    # tensor_parts = [x_stem_0, x_stem_1, x_stem_2, x_stem_3]
                    # print("tensor_parts", tensor_parts[0].shape)
                    # concat_tensor = [block_0(tensor_parts[i]) for i in range(len(tensor_parts))]
                    # concat_tensor = torch.cat(concat_tensor, dim=2)
                    # concat_tensor = block_1(concat_tensor)
                    # output_dict["stage%d" % stage_id] = concat_tensor
                    tensor_parts = [x_stem_0, x_stem_1, x_stem_2, x_stem_3]
                    concat_tensor = [stage(tensor_parts[i]) for i in range(len(tensor_parts))]
                    output_dict["stage%d" % stage_id] = concat_tensor
                elif stage_id ==2:
                    block_0= stage.op_list[0]
                    block_1= stage.op_list[1]
                    concat_tensor = [block_0(concat_tensor[i]) for i in range(len(concat_tensor))]
                    concat_tensor = torch.cat(concat_tensor, dim=2)
                    concat_tensor = block_1(concat_tensor)
                    output_dict["stage%d" % stage_id] = concat_tensor
                else:
                    for ops in stage:
                        concat_tensor = ops(concat_tensor)
                    output_dict["stage%d" % stage_id] = concat_tensor 
            output_dict["stage_final"] = concat_tensor
            return output_dict


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone

def efficientvit_backbone_b0_demo(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        branch=True,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone

def branch_efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        branch=True,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone

def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone

def efficientvit_backbone_b1_demo(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        branch=True,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone

def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone
