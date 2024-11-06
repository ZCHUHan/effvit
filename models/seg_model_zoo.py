from typing import Optional, Dict

from models.utils import load_state_dict_from_file
from models.efficientvit.seg import EfficientViTSeg
from models.efficientvit.quan_seg import Quan_EfficientViTSeg
__all__ = ["create_seg_model"]


REGISTERED_SEG_MODEL: Dict[str, Dict[str, str]] = {
    "cityscapes": {
        "b0-r960": "checkpoints/effvit/eff_b0.pt",
        "b1-r896": "assets/checkpoints/seg/cityscapes/b1-r896.pt",
        "b2-r1024": "assets/checkpoints/seg/cityscapes/b2-r1024.pt",
        "b3-r1184": "assets/checkpoints/seg/cityscapes/b3-r1184.pt",
    },
    "ade20k": {
        "b1-r480": "assets/checkpoints/seg/ade20k/b1-r480.pt",
        "b2-r416": "assets/checkpoints/seg/ade20k/b2-r416.pt",
        "b3-r512": "assets/checkpoints/seg/ade20k/b3-r512.pt",
    }
}


def create_seg_model(name: str, dataset: str, pretrained=True, weight_url: Optional[str] = None, **kwargs) -> EfficientViTSeg or Quan_EfficientViTSeg:
    from models.efficientvit import branch_efficientvit_seg_b0, efficientvit_seg_b0, efficientvit_seg_b0_demo, efficientvit_seg_b1, efficientvit_seg_b1_demo, efficientvit_seg_b2, efficientvit_seg_b3
    from models.efficientvit import quan_efficientvit_seg_b0, quan_efficientvit_seg_b0_demo, quan_efficientvit_seg_b1, quan_efficientvit_seg_b2, quan_efficientvit_seg_b3
    model_dict = {
        "b0": efficientvit_seg_b0,
        "b0_demo": efficientvit_seg_b0_demo,
        "branch_b0": branch_efficientvit_seg_b0,
        "b1": efficientvit_seg_b1,
        "b1_demo": efficientvit_seg_b1_demo,
        "b2": efficientvit_seg_b2,
        "b3": efficientvit_seg_b3,
        "quan_b0": quan_efficientvit_seg_b0,
        "quan_b0_demo": quan_efficientvit_seg_b0_demo,
        "quan_b1": quan_efficientvit_seg_b1,
        "quan_b2": quan_efficientvit_seg_b2,
        "quan_b3": quan_efficientvit_seg_b3,
    }

    model_id = name.split("-")[0]

    if model_id not in model_dict:
        print(model_id)
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](dataset=dataset, **kwargs)
    
    # if pretrained and weight_url is not None:
    #     weight_url = weight_url or REGISTERED_SEG_MODEL[dataset].get(name, None)
    #     try:
    #         weight = load_state_dict_from_file(weight_url)
    #         model.load_state_dict(weight)
    #     except RuntimeError as e:
    #         print(e)

    return model

