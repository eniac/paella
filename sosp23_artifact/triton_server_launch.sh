#!/bin/bash

sudo docker run -it --gpus=1 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 triton_server_tvm bash -c 'LD_LIBRARY_PATH=/workspace/src/tvm-tf/build:$LD_LIBRARY_PATH LD_PRELOAD="/workspace/src/tvm-tf/build/libtvm_dso_op.so /opt/tritonserver/backends/tensorflow2/libtensorflow_cc.so /opt/tritonserver/backends/tensorflow2/libtensorflow_framework.so" CUDA_DEVICE_MAX_CONNECTIONS=32 tritonserver --model-repository=/workspace/models/tensorflow --backend-config=tensorflow,version=2 --min-supported-compute-capability=6.0 --allow-grpc=true --backend-config=default-max-batch-size=0'

