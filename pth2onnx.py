from multiprocessing import dummy
import torch
import onnx
import os
from models.seg_model_zoo import create_seg_model
def pth_to_onnx(input, model, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    
    model.eval()
    model.to('cpu')
    # print(input.shape)
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=14) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")
os.environ['CUDA_VISIBLE_DEVICES']='0'
checkpoint = './checkpoints/effvit/eff_b0_branch.pth'
onnx_path = './onnx/effvit_b0_branch.onnx'
onnx_path_simp = './onnx/effvit_b0_branch_simp.onnx'
input = tuple([torch.randn(1, 3, 256, 256, requires_grad=False)])
input = torch.randn(1, 3, 256, 256, device='cpu')
# input = torch.randn(1, 3, 512, 512, device='cpu')

# model = create_seg_model('b0_demo', 'cityscapes') # w/ branch
model = create_seg_model('b0', 'cityscapes') # w/o branch

checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
_ = model(input)
# if "state_dict" in checkpoint:
#     checkpoint = checkpoint["state_dict"]
# else:
#     checkpoint = checkpoint["model_state"]
# try:
#     model.load_state_dict(checkpoint)    
# except RuntimeError as e:
#     print(e)
pth_to_onnx(input, model, onnx_path)

from onnxsim import simplify
onnx_model = onnx.load(onnx_path)
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_path_simp)
print("Simplified onnx model saved at {}".format(onnx_path))