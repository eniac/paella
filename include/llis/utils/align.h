#pragma once

#include <llis/utils/gpu.h>

#include <cstddef>
#include <cstdint>

namespace llis {
namespace utils {

CUDA_HOSTDEV inline size_t next_aligned_pos(size_t next_pos, size_t align) {
    return (next_pos + align - 1) & ~(align - 1);
}

template <typename T>
CUDA_HOSTDEV inline T* next_aligned_ptr(T* next_ptr, size_t align) {
    return reinterpret_cast<uintptr_t>(reinterpret_cast<char*>(next_ptr) + align - 1) & ~(align - 1);
}

}
}

