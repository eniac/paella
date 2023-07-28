#!/bin/bash

PREFIX=/bigdisk

cd "$(dirname "$0")"
abs_path="`pwd`"

# Get TVM source

cd "${PREFIX}/src"

git clone --recursive https://github.com/eniac/tvm-llis.git tvm-tf
cd tvm-tf
git checkout v0.10.0-llis
git submodule update --recursive

# Compile TVM for TF and Convert TVM models to TF models

sudo docker run --gpus=1 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
                -v${abs_path}/..:/workspace/sosp23_artifact \
                -v"${PREFIX}"/models:/workspace/models \
                -v"${PREFIX}"/src/tvm-tf:/workspace/src/tvm-tf \
                nvcr.io/nvidia/tensorflow:23.03-tf2-py3 \
                /workspace/sosp23_artifact/setup/triton_docker/run_on_tf_docker.sh # Use convert_tvm_to_tf.sh instead if building TVM is not necessary

sudo rsync -a ../tf_models_config/ ${PREFIX}/models/tensorflow/

# Build docker image

cd triton_docker

mkdir -p models
sudo mount --bind ${PREFIX}/models models

mkdir -p sosp23_artifact 
sudo mount --bind ../../../sosp23_artifact sosp23_artifact

mkdir -p tvm-tf
sudo mount --bind ${PREFIX}/src/tvm-tf tvm-tf

sudo docker build -t triton_server_tvm .

sudo umount models
sudo umount sosp23_artifact
sudo umount tvm-tf

