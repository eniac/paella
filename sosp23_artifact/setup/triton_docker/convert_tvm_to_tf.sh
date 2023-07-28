#!/bin/bash

export TVM_HOME=/workspace/src/tvm-tf
export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${TVM_HOME}/build:$LD_LIBRARY_PATH

apt update
apt install -y cmake clang

cd "$(dirname "$0")"
../dso_to_tf.py /workspace/models/cuda /workspace/models/tensorflow

