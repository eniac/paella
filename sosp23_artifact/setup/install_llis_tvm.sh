#!/bin/bash

PREFIX=/bigdisk
CUDA_ARCHITECTURES=60

sudo chown $USER "${PREFIX}"

mkdir -p "${PREFIX}/src"
mkdir -p "${PREFIX}/opt"

# Get LLIS source

cd "${PREFIX}/src"

#git clone https://github.com/eniac/llis.git # TODO: should use this
git clone https://github.com/eniac/paella.git llis
cd llis
git switch sosp23_artifact

# Get custom TVM source

cd "${PREFIX}/src"

git clone --recursive https://github.com/eniac/tvm-llis.git
cd tvm-llis
git checkout v0.10.0-llis
git submodule update --recursive

# Install LLIS

cd "${PREFIX}/src"

cd llis
mkdir release
cd release
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES -DTVM_PATH="${PREFIX}/src/tvm-llis" -DCMAKE_INSTALL_PREFIX=${PREFIX}/opt/llis
make -j40 install
echo "export CMAKE_PREFIX_PATH=${PREFIX}/opt/llis:\$CMAKE_PREFIX_PATH" | tee -a ~/.bash_profile
echo "export LD_LIBRARY_PATH=${PREFIX}/opt/llis/lib:\$LD_LIBRARY_PATH" | tee -a ~/.bash_profile
echo "export LIBRARY_PATH=${PREFIX}/opt/llis/lib:\$LIBRARY_PATH" | tee -a ~/.bash_profile
echo "export CPATH=${PREFIX}/opt/llis/include:\$CPATH" | tee -a ~/.bash_profile
echo "export CUDA_DEVICE_MAX_CONNECTIONS=32" | tee -a ~/.bash_profile

source /etc/profile
source ~/.bash_profile

# Install TVM

cd "${PREFIX}/src"

cd tvm-llis
mkdir build
cd build
cp ../cmake/config.cmake .
sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/' config.cmake
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/' config.cmake
echo "set(USE_LLIS ${PREFIX}/opt/llis)" >> config.cmake
cmake .. -DCMAKE_INSTALL_PREFIX=${PREFIX}/opt/tvm-llis
make -j40
make install
cd ../python
python setup.py install
cd ..
cp -r 3rdparty/dmlc-core/include/dmlc "${PREFIX}/opt/tvm-llis/include"
cp -r 3rdparty/dlpack/include/dlpack "${PREFIX}/opt/tvm-llis/include"
echo "export CMAKE_PREFIX_PATH=${PREFIX}/opt/tvm-llis:\$CMAKE_PREFIX_PATH" | tee -a ~/.bash_profile
echo "export LD_LIBRARY_PATH=${PREFIX}/opt/tvm-llis/lib:\$LD_LIBRARY_PATH" | tee -a ~/.bash_profile
echo "export LIBRARY_PATH=${PREFIX}/opt/tvm-llis/lib:\$LIBRARY_PATH" | tee -a ~/.bash_profile
echo "export CPATH=${PREFIX}/opt/tvm-llis/include:\$CPATH" | tee -a ~/.bash_profile

source /etc/profile
source ~/.bash_profile

# Install LLIS jobs and tests

cd "${PREFIX}/src"

cd llis/release
cmake .. -Utvm_FOUND # Find TVM again after we have installed it
make -j40 install

# Build models

cd "$(dirname "$0")"

./onnx2tvm_all.sh ${PREFIX}/models/onnx ${PREFIX}/models/cuda_llis cuda_llis
./onnx2tvm_all.sh ${PREFIX}/models/onnx ${PREFIX}/models/cuda cuda
rsync -a ../tvm_models_dim/ ${PREFIX}/models/cuda/

