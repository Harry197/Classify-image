import io
# import numpy as np

import torch.onnx
import timm
import argparse
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
parser.add_argument('--classes_path', required=True)
parser.add_argument('--export_path', type=str, default='./export/')
args = parser.parse_args()

os.makedirs(args.export_path, exist_ok=True)

if os.path.isdir(args.checkpoint_path):
    file_list = os.listdir(args.checkpoint_path)
    full_list = [os.path.join(args.checkpoint_path,i) for i in file_list]
    model_path = sorted(full_list, key=os.path.getmtime)[-1]
else:
    model_path = args.checkpoint_path

classes=[]
with open(args.classes_path, 'r') as csv_file:
    readers = csv.reader(csv_file)
    for reader in readers:
        classes.append(reader)

num_classes = len(classes)

model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes, drop_rate=0.2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

batch_size = 1

x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
torch_out = model(x)

torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  os.path.join(args.export_path, "model.onnx"),   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})