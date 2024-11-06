import torch.onnx

import torchvision
import os
import copy
import time
import argparse
from datetime import datetime
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn


import sys


import onnx, onnx.numpy_helper, onnx.helper
import numpy as np
from FLAT_pytorchnet_2_with_QuanConv_batch_PACT_manual_encoder_demo_p1 import Unet_MRM, Unet_MRM_decoder
import tensorflow as tf
from FLAT_pytorchnet_2_with_QuanConv_batch_PACT import Unet_MRM_Quan
def MRM_Quan_init_manual():

    model_ckpt = Unet_MRM_Quan()
    model_ckpt.load_state_dict(torch.load('E:/ckpt/good/model_Quan.pth'))


    # print(torch.from_numpy(weights['kpn_conv_1/kernel']).float().shape)
    model_Quan = Unet_MRM_decoder()

    # model_Quan.layer1_encoder_conv.weight.data = model_ckpt.layer1_encoder_conv.w.data
    # model_Quan.layer1_encoder_conv.bias.data =  model_ckpt.layer1_encoder_conv.b.data
    # model_Quan.layer2_encoder_conv.weight.data = model_ckpt.layer2_encoder_conv.w.data
    # model_Quan.layer2_encoder_conv.bias.data = model_ckpt.layer2_encoder_conv.b.data
    # model_Quan.layer3_encoder_conv.weight.data = model_ckpt.layer3_encoder_conv.w.data
    # model_Quan.layer3_encoder_conv.bias.data = model_ckpt.layer3_encoder_conv.b.data
    # model_Quan.layer4_encoder_conv.weight.data = model_ckpt.layer4_encoder_conv.w.data
    # model_Quan.layer4_encoder_conv.bias.data = model_ckpt.layer4_encoder_conv.b.data
    # model_Quan.layer5_encoder_conv.weight.data = model_ckpt.layer5_encoder_conv.w.data
    # model_Quan.layer5_encoder_conv.bias.data = model_ckpt.layer5_encoder_conv.b.data
    # model_Quan.layer6_encoder_conv.weight.data = model_ckpt.layer6_encoder_conv.w.data
    # model_Quan.layer6_encoder_conv.bias.data = model_ckpt.layer6_encoder_conv.b.data
    # model_Quan.layer7_encoder_conv.weight.data = model_ckpt.layer7_encoder_conv.w.data
    # model_Quan.layer7_encoder_conv.bias.data = model_ckpt.layer7_encoder_conv.b.data
    # model_Quan.layer8_encoder_conv.weight.data = model_ckpt.layer8_encoder_conv.w.data
    # model_Quan.layer8_encoder_conv.bias.data = model_ckpt.layer8_encoder_conv.b.data
    # model_Quan.layer9_encoder_conv.weight.data = model_ckpt.layer9_encoder_conv.w.data
    # model_Quan.layer9_encoder_conv.bias.data = model_ckpt.layer9_encoder_conv.b.data
    # model_Quan.layer10_encoder_conv.weight.data = model_ckpt.layer10_encoder_conv.w.data
    # model_Quan.layer10_encoder_conv.bias.data = model_ckpt.layer10_encoder_conv.b.data

    model_Quan.layer11_decoder_conv_transpose.weight.data = model_ckpt.layer11_decoder_conv_transpose.w.data
    model_Quan.layer11_decoder_conv_transpose.bias.data =  model_ckpt.layer11_decoder_conv_transpose.b.data
    model_Quan.layer12_decoder_conv.weight.data = model_ckpt.layer12_decoder_conv.w.data
    model_Quan.layer12_decoder_conv.bias.data = model_ckpt.layer12_decoder_conv.b.data
    model_Quan.layer13_decoder_conv_transpose.weight.data = model_ckpt.layer13_decoder_conv_transpose.w.data
    model_Quan.layer13_decoder_conv_transpose.bias.data = model_ckpt.layer13_decoder_conv_transpose.b.data
    model_Quan.layer14_decoder_conv_transpose.weight.data = model_ckpt.layer14_decoder_conv_transpose.w.data
    model_Quan.layer14_decoder_conv_transpose.bias.data = model_ckpt.layer14_decoder_conv_transpose.b.data
    model_Quan.layer15_decoder_conv_transpose.weight.data = model_ckpt.layer15_decoder_conv_transpose.w.data
    model_Quan.layer15_decoder_conv_transpose.bias.data = model_ckpt.layer15_decoder_conv_transpose.b.data
    model_Quan.layer16_decoder_conv.weight.data = model_ckpt.layer16_decoder_conv.w.data
    model_Quan.layer16_decoder_conv.bias.data = model_ckpt.layer16_decoder_conv.b.data
    model_Quan.layer17_decoder_conv_transpose.weight.data = model_ckpt.layer17_decoder_conv_transpose.w.data
    model_Quan.layer17_decoder_conv_transpose.bias.data = model_ckpt.layer17_decoder_conv_transpose.b.data
    model_Quan.layer18_decoder_conv_transpose.weight.data = model_ckpt.layer18_decoder_conv_transpose.w.data
    model_Quan.layer18_decoder_conv_transpose.bias.data = model_ckpt.layer18_decoder_conv_transpose.b.data
    model_Quan.layer19_decoder_conv_transpose.weight.data = model_ckpt.layer19_decoder_conv_transpose.w.data
    model_Quan.layer19_decoder_conv_transpose.bias.data = model_ckpt.layer19_decoder_conv_transpose.b.data
    model_Quan.layer20_decoder_conv.weight.data = model_ckpt.layer20_decoder_conv.w.data
    model_Quan.layer20_decoder_conv.bias.data = model_ckpt.layer20_decoder_conv.b.data

    model_Quan.layer21_decoder_conv_transpose.weight.data = model_ckpt.layer21_decoder_conv_transpose.w.data
    model_Quan.layer21_decoder_conv_transpose.bias.data =  model_ckpt.layer21_decoder_conv_transpose.b.data
    model_Quan.layer22_decoder_conv_transpose.weight.data = model_ckpt.layer22_decoder_conv_transpose.w.data
    model_Quan.layer22_decoder_conv_transpose.bias.data = model_ckpt.layer22_decoder_conv_transpose.b.data
    model_Quan.layer23_decoder_conv_transpose.weight.data = model_ckpt.layer23_decoder_conv_transpose.w.data
    model_Quan.layer23_decoder_conv_transpose.bias.data = model_ckpt.layer23_decoder_conv_transpose.b.data

    model_Quan.layer24_decoder_conv.weight.data =  model_ckpt.layer24_decoder_conv.w.data
    model_Quan.layer24_decoder_conv.bias.data =  model_ckpt.layer24_decoder_conv.b.data


    model_Quan.layer25_decoder_conv.weight.data =  model_ckpt.layer25_decoder_conv.w.data
    model_Quan.layer25_decoder_conv.bias.data =  model_ckpt.layer25_decoder_conv.b.data

    model_Quan.layer26_decoder_conv.weight.data =  model_ckpt.layer26_decoder_conv.w.data
    model_Quan.layer26_decoder_conv.bias.data =  model_ckpt.layer26_decoder_conv.b.data

    return model_Quan

def Convert_ONNX(net, name): 

    # set the model to inference mode 
    net.eval() 
    net = net.to("cpu")

    # Let's create a dummy input tensor  
    dummy_input = []
    
    dummy_input_1 = torch.randn(1, 9, 224, 224, requires_grad=True)  
    dummy_input_2 = torch.randn(1, 512, 14, 14, requires_grad=True)  
    dummy_input_3 = torch.randn(1, 256, 28, 28, requires_grad=True)  
    dummy_input_4 = torch.randn(1, 128, 56, 56, requires_grad=True)     
    dummy_input_5 = torch.randn(1, 64, 112, 112, requires_grad=True)    

    dummy_input = tuple([dummy_input_1, dummy_input_2, dummy_input_3, dummy_input_4, dummy_input_5])

    # Export the model   
    torch.onnx.export(net,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         name + ".onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}},
          training=torch.onnx.TrainingMode.EVAL) 
    print(" ") 
    print('Model has been converted to ONNX') 
    
if __name__ == '__main__':

    model_manual_quan = MRM_Quan_init_manual()

    Convert_ONNX(model_manual_quan,'E:\pytorch_FLAT\model_manual_quan_onnx_demo_decoder')

    # dummy_input_1 = torch.randn(1, 9, 224, 224, requires_grad=True)  
    # output = model_manual_quan(dummy_input_1)

