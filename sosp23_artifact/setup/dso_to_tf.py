#!/usr/bin/python3

import os
import shutil
import tensorflow as tf
import argparse
import tvm
from tvm.contrib import tf_op

class MyModel(tf.Module):
    def __init__(self, so_path, out_shape, out_type):
        module = tf_op.OpGraphModule(so_path)
        self.op = module.graph(output_shape=out_shape, output_dtype=out_type)
        self.out_shape = out_shape
        self.out_type = out_type

    @tf.function
    def serve(self, x):
        return tf.reshape(self.op(x), self.out_shape)


parser = argparse.ArgumentParser()
parser.add_argument('import_dir', type=str)
parser.add_argument('export_dir', type=str)
args = parser.parse_args()
print("Converting {} to {}...".format(args.import_dir, args.export_dir))


models = [os.path.join(r, fn) for r, ds, fs in os.walk(args.import_dir) for fn in fs if fn.endswith(".so")]
print("Found models")
print(models)

vision_metadata = {
    'in_name': 'input_1', 'in_shape': [1, 3, 224, 224], 'in_type': tf.float32,
    'out_name': 'output_0', 'out_shape': [1, 1000], 'out_type': tf.float32
}

info = {
    'mnist-8': {
        'in_name': 'input_1', 'in_shape': [1, 1, 28, 28], 'in_type': tf.float32,
        'out_name': 'output_0', 'out_shape': [1, 10], 'out_type': tf.float32,
    },
    'densenet-9': {
        'in_name': 'input_1', 'in_shape': [1, 3, 224, 224], 'in_type': tf.float32,
        'out_name': 'output_0', 'out_shape': [1, 1000, 1, 1], 'out_type': tf.float32,
    },
    'googlenet-9': vision_metadata,
    'inception_v3': vision_metadata,
    'mobilenetv2-7': vision_metadata,
    'resnet18-v2-7': vision_metadata,
    'resnet34-v2-7': vision_metadata,
    'resnet50-v2-7': vision_metadata,
    'squeezenet1.1-7': vision_metadata,
}

for m in models:
    model_name = m.split('/')[-1].rstrip('cuda-pack.so')
    name = None
    metadata = None
    for k, v in info.items():
        if k == model_name:
            name = k
            metadata = info[name]
            break

    if name is None:
        print("No metadata for model {}".format(m))
        continue

    print("Converting {}...".format(name))

    export_path = os.path.join(args.export_dir, name, "1/model.savedmodel")
    shutil.rmtree(export_path, ignore_errors=True)

    export_model = MyModel(m, metadata["out_shape"], metadata["out_type"])

    tf.saved_model.save(export_model, export_path, signatures=export_model.serve.get_concrete_function(
        tf.TensorSpec(shape=metadata['in_shape'], dtype=metadata['in_type'], name='input_1')
    ))
