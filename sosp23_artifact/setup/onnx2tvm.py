import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import sys

model_path = sys.argv[1] # path to the onnx model
output_path = sys.argv[2] # path to output the tvm-compiled model
target = sys.argv[3] # cuda or cuda -llis_flag=[1|3|5]
input_dims = tuple(int(x) for x in sys.argv[4:]) # e.g., `1 3 224 224` for mobilenet

onnx_model = onnx.load(model_path)

input_names_all = [node.name for node in onnx_model.graph.input]
input_initializer =  [node.name for node in onnx_model.graph.initializer]
input_names = list(set(input_names_all)  - set(input_initializer))

input_name = input_names[0]

shape_dict = {input_name: input_dims}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

opt_level = 3
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

lib.export_library(output_path)

