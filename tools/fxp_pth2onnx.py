import torch
import torch.nn as nn
import onnx
import onnx.helper as helper
import os
from models.seg_model_zoo import create_seg_model
import argparse

# Step 1: Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='quan_b0', help='Model name')
parser.add_argument('--weight_bit', type=int, default=8, help='Weight quantization bit-width')
parser.add_argument('--activation_bit', type=int, default=8, help='Activation quantization bit-width')
args = parser.parse_args()

# Step 2: Load the quantized model
os.environ['CUDA_VISIBLE_DEVICES']='0'
checkpoint = '../checkpoints/dyadic_hswish_wo_psumquan/best_dyadic.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
quantized_model = create_seg_model(args.model_name, 'cityscapes')
checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
else:
    checkpoint = checkpoint["model_state"]
try:
    quantized_model.load_state_dict(checkpoint)    
except RuntimeError as e:
    print(e)

# Step 3: Convert the quantized PyTorch model to an ONNX model and save it to a file
quantized_model = nn.DataParallel(quantized_model)
quantized_model.to(device)
quantized_model.eval()
dummy_input = torch.randn(1, 3, 1024, 1920)
onnx_model_path = f"./onnx/{args.model_name}_quantized.onnx"
torch.onnx.export(quantized_model, dummy_input, onnx_model_path, opset_version=12)

# Step 4: Modify the ONNX model to include scale and bit-depth information for each node
model = onnx.load(onnx_model_path)
graph = model.graph

for node in graph.node:
    # Add weight bit-width attribute
    weight_bit_attribute = helper.make_attribute("weight_bit", args.weight_bit)
    node.attribute.append(weight_bit_attribute)

    # Add activation bit-width attribute
    activation_bit_attribute = helper.make_attribute("activation_bit", args.activation_bit)
    node.attribute.append(activation_bit_attribute)

# Step 5: Save the modified ONNX model
modified_onnx_model_path = f"./onnx/{args.model_name}_quantized_modified.onnx"
onnx.save(model, modified_onnx_model_path)
