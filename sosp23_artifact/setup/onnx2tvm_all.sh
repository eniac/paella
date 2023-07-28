#!/bin/bash

ONNX_DIR=$1
DEST_DIR=$2
TYPE=$3

if [[ $TYPE == "cuda_llis" ]]; then
    TYPE2='cuda -llis_flag=3'
else
    TYPE2='cuda'
fi

mkdir -p $DEST_DIR

python3 onnx2tvm.py ${ONNX_DIR}/mnist-8.onnx ${DEST_DIR}/mnist-8-${TYPE}-pack.so "${TYPE2}" 1 1 28 28
python3 onnx2tvm.py ${ONNX_DIR}/densenet-9.onnx ${DEST_DIR}/densenet-9-${TYPE}-pack.so "${TYPE2}" 1 3 224 224
python3 onnx2tvm.py ${ONNX_DIR}/googlenet-9.onnx ${DEST_DIR}/googlenet-9-${TYPE}-pack.so "${TYPE2}" 1 3 224 224
python3 onnx2tvm.py ${ONNX_DIR}/mobilenetv2-7.onnx ${DEST_DIR}/mobilenetv2-7-${TYPE}-pack.so "${TYPE2}" 1 3 224 224
python3 onnx2tvm.py ${ONNX_DIR}/resnet18-v2-7.onnx ${DEST_DIR}/resnet18-v2-7-${TYPE}-pack.so "${TYPE2}" 1 3 224 224
python3 onnx2tvm.py ${ONNX_DIR}/resnet34-v2-7.onnx ${DEST_DIR}/resnet34-v2-7-${TYPE}-pack.so "${TYPE2}" 1 3 224 224
python3 onnx2tvm.py ${ONNX_DIR}/resnet50-v2-7.onnx ${DEST_DIR}/resnet50-v2-7-${TYPE}-pack.so "${TYPE2}" 1 3 224 224
python3 onnx2tvm.py ${ONNX_DIR}/squeezenet1.1-7.onnx ${DEST_DIR}/squeezenet1.1-7-${TYPE}-pack.so "${TYPE2}" 1 3 224 224
python3 onnx2tvm.py ${ONNX_DIR}/inception_v3.onnx ${DEST_DIR}/inception_v3-${TYPE}-pack.so "${TYPE2}" 1 3 224 224
