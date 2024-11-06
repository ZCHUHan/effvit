from typing import List, Optional
from sympy import per

import torch
import torch.nn as nn

from models.utils import build_kwargs_from_config
from models.nn import OpSequential, ConvLayer, DAGBlock, UpSampleLayer, MBConv, ResidualBlock, IdentityLayer
from models.nn import QUpSampleLayer, QConvLayer, QResidualBlock, QEfficientViTBlock, QDSConv, QMBConv
from models.efficientvit.quan_backbone import Quan_EfficientViTBackbone
from models.nn.quant_lsq import QuanConv, PActFn, PACT, SymmetricQuantFunction
__all__ = [
    "Quan_EfficientViTSeg",
    "quan_efficientvit_seg_b0",
    "quan_efficientvit_seg_b0_demo",
    "quan_efficientvit_seg_b1",
    "quan_efficientvit_seg_b2",
    "quan_efficientvit_seg_b3",
]

nbit_w = 8
nbit_a = 8
quan_w = 'lsq'
quan_a = 'lsq'
per_channel = True
class SegHead(DAGBlock):
    def __init__(self, fid_list: List[str], in_channel_list: List[int], stride_list: List[int], head_stride: int, head_width: int, head_depth: int, expand_ratio: float, final_expand: Optional[float], n_classes: int, dropout_rate=0, norm="bn2d", act_func="quanhswish"):
        inputs = {}
        self.n_classes = n_classes
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            
            if factor == 1:
                inputs[fid] = QConvLayer(
                    in_channel,
                    head_width,
                    1,
                    per_channel=per_channel,
                    norm=norm,
                    act_func=None,
                    nbit_w=nbit_w,
                    nbit_a=nbit_a,
                    quan_a=quan_a,
                    quan_w=quan_w)
            else:
                inputs[fid] = OpSequential([
                    QConvLayer(
                        in_channel,
                        head_width,
                        1,
                        per_channel=per_channel,
                        norm=norm,
                        act_func=None,
                        nbit_w=nbit_w,
                        nbit_a=nbit_a,
                        quan_a=quan_a,
                        quan_w=quan_w
                        ),
                    QUpSampleLayer(factor=factor, quan_a_name=quan_a, nbit_a=nbit_a),
                    # UpSampleLayer(size=size),
                ])
        middle = []
        for _ in range(head_depth):
            block = QMBConv(
                head_width,                 # 32
                head_width,                 # 32
                per_channel=per_channel,
                expand_ratio=expand_ratio,  # 4
                norm=norm, 
                act_func=(act_func, act_func, None),
                nbit_w=nbit_w,
                nbit_a=nbit_a,
                quan_a=quan_a,
                quan_w=quan_w,
                res=True
            )
            middle.append(block)
        middle = OpSequential(middle)

        outputs = {
            "segout": OpSequential([
                None if final_expand is None else
                QConvLayer(head_width,
                           head_width * final_expand,
                           1,
                           per_channel=per_channel,
                           norm=norm,
                           act_func=act_func,
                           nbit_a=nbit_a,
                           nbit_w=nbit_w,
                           quan_a=quan_a,
                           quan_w=quan_w),
                QConvLayer(
                    head_width * (final_expand or 1), 
                    n_classes, 
                    1, 
                    per_channel = per_channel,
                    use_bias=True, 
                    dropout_rate=dropout_rate, 
                    norm=None, 
                    act_func=None,
                    nbit_w=nbit_w,
                    nbit_a=nbit_a,
                    quan_a=quan_a,
                    quan_w=quan_w
                )
            ])
        }
        
        super(SegHead, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)



class Quan_EfficientViTSeg(nn.Module):
    def __init__(self, backbone: Quan_EfficientViTBackbone, head: SegHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def get_psum_loss(self, model, loss, loss_weight=0.1):
        children = list(model.named_children())
        for name, child in children:
            if isinstance(child, QuanConv):
                if child.psum_quan:
                    # print("name: ", name)
                    # print("psum loss: ", child.psum_loss)
                    loss += (child.psum_loss * loss_weight)
            else:
                self.get_psum_loss(child, loss)
            
    def forward(self, x: torch.Tensor, fix_bn=False) -> torch.Tensor:
        feed_dict = self.backbone(x, fix_bn)
        
        self.head.input_ops[2].conv.lsq_a = self.backbone.stages[2][0].inverted_conv.conv.lsq_a
        self.head.input_ops[1].op_list[0].conv.lsq_a = self.backbone.stages[3][0].inverted_conv.conv.lsq_a
        
        feed_dict = self.head(feed_dict)
        loss_l2 = torch.tensor(0.0, device=x.device)
        self.get_psum_loss(self, loss_l2)
        return feed_dict["segout"]


def quan_efficientvit_seg_b0(dataset: str, **kwargs) -> Quan_EfficientViTSeg:
    from models.efficientvit.quan_backbone import quan_efficientvit_backbone_b0
    backbone = quan_efficientvit_backbone_b0(**kwargs)
    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = Quan_EfficientViTSeg(backbone, head)
    return model

def quan_efficientvit_seg_b0_demo(dataset: str, **kwargs) -> Quan_EfficientViTSeg:
    from models.efficientvit.quan_backbone import quan_efficientvit_backbone_b0_demo
    backbone = quan_efficientvit_backbone_b0_demo(**kwargs)
    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = Quan_EfficientViTSeg(backbone, head)
    return model

def quan_efficientvit_seg_b1(dataset: str, **kwargs) -> Quan_EfficientViTSeg:
    from models.efficientvit.quan_backbone import quan_efficientvit_backbone_b1
    backbone = quan_efficientvit_backbone_b1(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[256, 128, 64],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=64,
            head_depth=3,
            expand_ratio=4,
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[256, 128, 64],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=64,
            head_depth=3,
            expand_ratio=4,
            final_expand=None,
            n_classes=150,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = Quan_EfficientViTSeg(backbone, head)
    return model


def quan_efficientvit_seg_b2(dataset: str, **kwargs) -> Quan_EfficientViTSeg:
    from models.efficientvit.quan_backbone import quan_efficientvit_backbone_b2
    backbone = quan_efficientvit_backbone_b2(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[384, 192, 96],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=96,
            head_depth=3,
            expand_ratio=4,
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[384, 192, 96],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=96,
            head_depth=3,
            expand_ratio=4,
            final_expand=None,
            n_classes=150,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = Quan_EfficientViTSeg(backbone, head)
    return model


def quan_efficientvit_seg_b3(dataset: str, **kwargs) -> Quan_EfficientViTSeg:
    from models.efficientvit.quan_backbone import quan_efficientvit_backbone_b3
    backbone = quan_efficientvit_backbone_b3(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            final_expand=None,
            n_classes=150,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = Quan_EfficientViTSeg(backbone, head)
    return model

