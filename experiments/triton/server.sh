#! /bin/bash -ex

MODELS_PATH=$1 # path to TF-wrapped models
LIBTVM_PATH=$2 # path to our TVM
CMAKE_PATH=$3 # path to our cmake
LLIS_PATH=$4 # path to ALLIS libraries
LIBBOOST_PATH=$5

docker run -it --gpus=1 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v${MODELS_PATH}:/models -v${LIBTVM_PATH}:/opt/tvm -v${CMAKE_PATH}:/opt/cmake -v${LLIS_PATH}:/opt/allis -v${LIBBOOST_PATH}:/opt/boost nvcr.io/nvidia/tritonserver:latest
