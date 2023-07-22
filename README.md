# LLIS

## Build

```
mkdir build
cmake -DCMAKE_BUILD_TYPE=<release|debug> -DCMAKE_CUDA_ARCHITECTURES=<cuda_arch> -DTVM_PATH=/path/to/tvm-llis/source ..
make -j$(nproc)
```

