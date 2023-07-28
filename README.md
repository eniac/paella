# Paella / LLIS

This project was called LLIS at the very beginning, and so this name is used in the codebase.

## SOSP 2023 Artifact Evaluation

Please refer to the [instructions](sosp23_artifact/README.md) in the `sosp23_artifact/` directory.

## Build

```
mkdir build
cmake -DCMAKE_BUILD_TYPE=<release|debug> -DCMAKE_CUDA_ARCHITECTURES=<cuda_arch> -DTVM_PATH=/path/to/tvm-llis/source ..
make -j$(nproc)
```

