#!/bin/bash

apt update
apt install -y cmake clang

cd /workspace/src/tvm-tf

mkdir build
cd build
cp ../cmake/config.cmake .
sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/' config.cmake
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' config.cmake
sed -i 's/set(USE_TF_TVMDSOOP OFF)/set(USE_TF_TVMDSOOP ON)/' config.cmake
cmake ..
make -j40

