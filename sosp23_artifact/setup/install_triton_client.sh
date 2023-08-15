#!/bin/bash

PREFIX=/bigdisk

while getopts 'p:' opt; do
  case "$opt" in
    p)
      PREFIX="$OPTARG"
      ;;

    ?|h)
      echo "Usage: $(basename $0) [-p PREFIX]"
      exit 1
      ;;
  esac
done

source /etc/profile
source ~/.bash_profile

sudo chown $USER "${PREFIX}"

mkdir -p "${PREFIX}/src"

cd "${PREFIX}/src"

sudo apt-get install -y curl libcurl4-openssl-dev libb64-dev libssl-dev zlib1g-dev rapidjson-dev libopencv-dev libyaml-cpp-dev
pip install PyYAML

git clone https://github.com/maxdml/triton-client.git triton-client-llis
cd triton-client-llis
git checkout sosp23_artifact
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PERF_ANALYZER=OFF -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_ENABLE_TESTS=ON ..
make -j40

