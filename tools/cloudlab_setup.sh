#!/bin/bash

PREFIX=/bigdisk

CMAKE_VERSION=3.19.0
GCC_VERSION=10.2.0

sudo chown kelvinng $PREFIX
mkdir -p $PREFIX/src
mkdir -p $PREFIX/opt
cd $PREFIX/src

# Install CUDA and NVIDIA driver

wget 'https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run'
sudo modprobe -r nouveau
sudo sh cuda_11.1.0_455.23.05_linux.run --silent --driver --toolkit --no-opengl-libs --installpath=$PREFIX/opt/cuda-11.1
echo "export CUDA_PATH=$PREFIX/opt/cuda-11.1" | sudo tee /etc/profile.d/cuda.sh
echo "export PATH=$PREFIX/opt/cuda-11.1/bin:\$PATH" | sudo tee -a /etc/profile.d/cuda.sh
rm cuda_11.1.0_455.23.05_linux.run

sudo apt install -y openjdk-8-jre

# Install CuDNN

tar -xf /proj/pennnetworks-PG0/cuda_installer/cudnn-11.1-linux-x64-v8.0.4.30.tgz -C .
sudo cp cuda/include/cudnn*.h $PREFIX/opt/cuda-11.1/include
sudo cp cuda/lib64/libcudnn* $PREFIX/opt/cuda-11.1/lib64
sudo chmod a+r $PREFIX/opt/cuda-11.1/include/cudnn*.h $PREFIX/opt/cuda-11.1/lib64/libcudnn*
rm -r cuda

# Install CMake

wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz"
tar -xf cmake-${CMAKE_VERSION}.tar.gz
cd cmake-${CMAKE_VERSION}
./bootstrap --prefix=$PREFIX/opt/cmake-${CMAKE_VERSION}
make -j8
make install
echo "export PATH=$PREFIX/opt/cmake-${CMAKE_VERSION}/bin:\$PATH" | tee -a ~/.bash_profile
cd $PREFIX/src
rm cmake-${CMAKE_VERSION}.tar.gz

# Install Miniconda

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $PREFIX/opt/miniconda3
echo "source $PREFIX/opt/miniconda3/etc/profile.d/conda.sh" | sudo tee /etc/profile.d/miniconda3.sh
echo 'conda activate base' | sudo tee -a /etc/profile.d/miniconda3.sh
rm Miniconda3-latest-Linux-x86_64.sh
source /etc/profile

# Install docker and nvidia-docker

curl https://get.docker.com | sudo sh
sudo systemctl enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo sed -i "1 a \ \ \ \ \"data-root\":\"$PREFIX/docker\"," /etc/docker/daemon.json
sudo systemctl start docker

sudo usermod -aG docker kelvinng

# Install Triton

sudo docker pull nvcr.io/nvidia/tritonserver:20.09-py3
sudo docker pull nvcr.io/nvidia/tritonserver:20.09-py3-clientsdk

# Get example Triton models #and build client Docker image

mkdir -p $PREFIX/src/triton
cd $PREFIX/src/triton
git clone https://github.com/triton-inference-server/server.git
cd server
git checkout r20.09
#sudo docker build -t tritonserver_client -f Dockerfile.client .
cd docs/examples
./fetch_models.sh
cd $PREFIX/src

mkdir -p $PREFIX/opt/triton-tools/bin
cat << EOF > $PREFIX/opt/triton-tools/bin/triton_docker.sh
#!/bin/bash

sudo docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$PREFIX/src/triton/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:20.09-py3 tritonserver --model-repository=/models
EOF
chmod +x $PREFIX/opt/triton-tools/bin/triton_docker.sh

cat << EOF > $PREFIX/opt/triton-tools/bin/triton_clientsdk_docker.sh
#!/bin/bash

sudo docker run -it --rm --net=host -v/:/host nvcr.io/nvidia/tritonserver:20.09-py3-clientsdk
EOF
chmod +x $PREFIX/opt/triton-tools/bin/triton_clientsdk_docker.sh

echo "export PATH=$PREFIX/opt/triton-tools/bin:\$PATH" | sudo tee /etc/profile.d/triton.sh

# Install some tools for ML

conda install -c conda-forge onnx

# Install Clipper and PyTorch

conda create --name clipper -y python=3.7
conda activate clipper

conda install -y pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

pip install git+https://github.com/ucbrise/clipper.git@develop#subdirectory=clipper_admin

conda deactivate

# Install TF Serving

pip install tensorflow

sudo docker pull tensorflow/serving:latest-gpu
mkdir tensorflow
cd tensorflow
git clone https://github.com/tensorflow/serving
cd $PREFIX/src

