import netron

import torch.onnx

from torch.autograd import Variable
from models import EMGClassifierV1
from utils import parse_args
import models
import os
import onnx
import onnx.utils
import onnx.version_converter

config = parse_args()
data = torch.randn(64, 8, 128, 1024)

model_class = getattr(models, config.models)
model = model_class()
# 导出 ONNX 模型
if not os.path.exists('./view_model'):
    os.makedirs('./view_model')
    
    
onnx_path = f"./view_model/{config.models}.onnx"
#make sure there is a view_model folder

torch.onnx.export(model, data, onnx_path, export_params= True,verbose=True)# opset_version 是onnx的版本


onnx_model = onnx.load(onnx_path)
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), onnx_path)
