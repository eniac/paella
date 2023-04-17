# Overview

We use triton 23:03 to run our experiments. We configure it to use our TVM, a dependency for our models.
We modify triton existing sample clients to use the same load generation logic we use across the paper.

dependencies
---
- Triton and our models need the same TVM
- Triton and TVM need the same tensorflow

# Server

Run `./server.sh [models path] [TVM lib path] [ALLIS lib path] [CMake path]`. This is the command we ran for our experiments:
```
./server.sh /home/maxdml/allis/models /home/maxdml/tvm_tf/ /home/kelvin/opt/cmake-3.22.3/ /home/kelvin/llis/ /home/kelvin/opt/boost-1.74.0/
```

In the container, run:
```
LD_PRELOAD="/opt/tvm/build/libtvm_dso_op.so /opt/tritonserver/backends/tensorflow2/libtensorflow_cc.so /opt/tritonserver/backends/tensorflow2/libtensorflow_framework.so" tritonserver --model-repository=/models/newmix3/tensorflow --backend-config=tensorflow,version=2 --min-supported-compute-capability=7.5 --allow-grpc=true --backend-config=default-max-batch-size=0
```

# Client

We use a custom client using triton's client framework.

Build
---
```
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PERF_ANALYZER=ON -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_ENABLE_TESTS=ON ..
```

Running
---
- We use run.py to run experiments
- run.py takes a config file describing the workload
