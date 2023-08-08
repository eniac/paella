# Paella / LLIS

This project was called LLIS at the very beginning, and so this name is used in the codebase.

## SOSP 2023 Artifact Evaluation

Please refer to the [instructions](sosp23_artifact/README.md) in the `sosp23_artifact/` directory.

## Dependencies

1. Linux (tested on Ubuntu 22.04)
1. NVIDIA driver (tested on 535.54.03)
1. CUDA runtime (tested on 12.2.0)
1. GCC (tested on 11.3.0)
1. CMake (tested on 3.22.1)
1. Boost (tested on 1.82.0)
1. LLVM / Clang (tested on 14)
1. spdlog (tested on 1.11.0; 1.12.0 is known to not work)
1. [**tvm-llis**](https://github.com/eniac/tvm-llis) (Custom version of TVM modified to work with Paella)

## Installation

### Paella/LLIS server and libraries

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=<release|debug> -DCMAKE_CUDA_ARCHITECTURES=<cuda_arch> .. # cuda_arch is 60 for 6.0, 75 for 7.5, etc
make -j$(nproc) install
```

### Custom TVM (tvm-llis)

Custom TVM depends on the libraries of Paella/LLIS. So, it can only be built after doing the previous step.

Please refer to [README-llis.md](https://github.com/eniac/tvm-llis/blob/v0.10.0-llis/README-llis.md) of [tvm-llis](https://github.com/eniac/tvm-llis) for instructions.

### Paella/LLIS applications (e.g., client) and job adapters

Applications and job adapters depend on the custom TVM. So, they can only be built after doing the previous step.

```
cmake .. -Utvm_FOUND # Find TVM again after we have installed it
make -j$(nproc) install
```

