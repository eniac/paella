#!/bin/bash

PREFIX=/workspace

apt update
apt install -y clang

echo "export LD_LIBRARY_PATH=${PREFIX}/src/tvm-tf/build:\$LD_LIBRARY_PATH" | tee -a ~/.bashrc

