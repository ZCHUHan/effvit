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
    # model.to(device)
    
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")
os.environ['CUDA_VISIBLE_DEVICES']='0'
checkpoint = './checkpoints/effvit/quan_b0_onnx.pth'
onnx_path = './effvit_b0.onnx'
onnx_path_simp = './effvit_b0_simp.onnx'
input = torch.randn(1, 3, 1024, 1920)
model = create_seg_model('b0-r960', 'cityscapes')
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
else:
    checkpoint = checkpoint["model_state"]
try:
    model.load_state_dict(checkpoint)    
except RuntimeError as e:
    print(e)
pth_to_onnx(input, model, onnx_path)
from onnxsim import simplify
onnx_model = onnx.load(onnx_path)
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_path_simp)
print("Simplified onnx model saved at {}".format(onnx_path))