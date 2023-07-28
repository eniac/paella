FROM nvcr.io/nvidia/tritonserver:23.03-py3

COPY sosp23_artifact /workspace/sosp23_artifact
COPY models/cuda /workspace/models/cuda
COPY models/tensorflow /workspace/models/tensorflow
COPY tvm-tf /workspace/src/tvm-tf

RUN /workspace/sosp23_artifact/setup/triton_docker/setup.sh

