#!/bin/bash

PREFIX=/bigdisk

CUDA_VERSION=12.2.0
NVIDIA_VERSION=535.54.03
BOOST_VERSION=1.82.0
SPDLOG_VERSION=1.11.0 # 1.12.0 does not work

sudo mkdir -p "${PREFIX}"
sudo chown $USER "${PREFIX}"

mkdir -p "${PREFIX}/src"
mkdir -p "${PREFIX}/opt"

# Install CUDA and NVIDIA driver

cd "${PREFIX}/src"

wget "https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda_${CUDA_VERSION}_${NVIDIA_VERSION}_linux.run"
sudo modprobe -r nouveau
sudo sh cuda_${CUDA_VERSION}_${NVIDIA_VERSION}_linux.run --silent --driver --toolkit --no-opengl-libs --installpath=$PREFIX/opt/cuda-${CUDA_VERSION}
echo "export CUDA_PATH=${PREFIX}/opt/cuda-${CUDA_VERSION}" | tee -a ~/.bash_profile
echo "export PATH=${PREFIX}/opt/cuda-${CUDA_VERSION}/bin:\$PATH" | tee -a ~/.bash_profile
echo "export LD_LIBRARY_PATH=${PREFIX}/opt/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH" | tee -a ~/.bash_profile

source /etc/profile
source ~/.bash_profile

# Install Miniconda

cd "${PREFIX}/src"

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $PREFIX/opt/miniconda3
echo "source $PREFIX/opt/miniconda3/etc/profile.d/conda.sh" | tee -a ~/.bash_profile
echo 'conda activate base' | tee -a ~/.bash_profile

source /etc/profile
source ~/.bash_profile

# Install docker and nvidia-docker

cd "${PREFIX}/src"

curl https://get.docker.com | sudo sh
sudo systemctl enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Install TVM dependencies

cd "${PREFIX}/src"

pip3 install numpy decorator attrs typing-extensions psutil scipy tornado onnx pandas matplotlib
mv ${PREFIX}/opt/miniconda3/lib/libstdc++.so.6{,.bak} # The libstdc++ provided by conda is too old. Force using the system one

# Install Boost

cd "${PREFIX}/src"

wget "https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION//\./_}.tar.gz"
tar -xf "boost_${BOOST_VERSION//\./_}.tar.gz"
cd "boost_${BOOST_VERSION//\./_}"
./bootstrap.sh --prefix="${PREFIX}/opt/boost-${BOOST_VERSION}"
./b2 install
echo "export CMAKE_PREFIX_PATH=${PREFIX}/opt/boost-${BOOST_VERSION}:\$CMAKE_PREFIX_PATH" | tee -a ~/.bash_profile
echo "export LD_LIBRARY_PATH=${PREFIX}/opt/boost-${BOOST_VERSION}/lib:\$LD_LIBRARY_PATH" | tee -a ~/.bash_profile
echo "export LIBRARY_PATH=${PREFIX}/opt/boost-${BOOST_VERSION}/lib:\$LIBRARY_PATH" | tee -a ~/.bash_profile
echo "export CPATH=${PREFIX}/opt/boost-${BOOST_VERSION}/include:\$CPATH" | tee -a ~/.bash_profile

source /etc/profile
source ~/.bash_profile

# Install clang

sudo apt install -y clang

# Install cmake

sudo apt install -y cmake

# Install spdlog

cd "${PREFIX}/src"

wget https://github.com/gabime/spdlog/archive/refs/tags/v${SPDLOG_VERSION}.tar.gz -O spdlog-${SPDLOG_VERSION}.tar.gz
tar -xf spdlog-${SPDLOG_VERSION}.tar.gz
cd spdlog-${SPDLOG_VERSION}
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX}/opt/spdlog-${SPDLOG_VERSION}
make -j40
make install
echo "export CMAKE_PREFIX_PATH=${PREFIX}/opt/spdlog-${SPDLOG_VERSION}:\$CMAKE_PREFIX_PATH" | tee -a ~/.bash_profile
echo "export LD_LIBRARY_PATH=${PREFIX}/opt/spdlog-${SPDLOG_VERSION}/lib:\$LD_LIBRARY_PATH" | tee -a ~/.bash_profile
echo "export LIBRARY_PATH=${PREFIX}/opt/spdlog-${SPDLOG_VERSION}/lib:\$LIBRARY_PATH" | tee -a ~/.bash_profile
echo "export CPATH=${PREFIX}/opt/spdlog-${SPDLOG_VERSION}/include:\$CPATH" | tee -a ~/.bash_profile

source /etc/profile
source ~/.bash_profile

## TODO: Triton
#
#cd "${PREFIX}/src"
