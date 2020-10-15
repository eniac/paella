#pragma once

// From https://stackoverflow.com/a/32015007/1213644
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

