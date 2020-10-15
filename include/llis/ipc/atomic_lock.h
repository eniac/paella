#pragma once

#include <llis/utils/gpu.h>

#include <atomic>

template <bool for_gpu>
class AtomicLock {};

template <>
class AtomicLock<false> {
  public:
    inline void acquire() {
        while (val_.test_and_set(std::memory_order_acquire));
    }

    inline void release() {
        val_.clear(std::memory_order_release);
    }

    inline void init() {
        release();
    }
    
  private:
    std::atomic_flag val_;
};

template <>
class AtomicLock<true> {
  public:
    CUDA_HOSTDEV inline void acquire() {
#ifdef __CUDA_ARCH__
        while (atomicOr(&val_gpu_, 1));
#else
        while (val_cpu_.test_and_set(std::memory_order_acquire));
#endif
    }

    CUDA_HOSTDEV inline void release() {
#ifdef __CUDA_ARCH__
        val_gpu_ = 0;
#else
        val_cpu_.clear(std::memory_order_release);
#endif
    }

    inline void init() {
        *reinterpret_cast<volatile unsigned int*>(&val_gpu_) = 0;
    }

  private:
    union {
        unsigned int val_gpu_;
        std::atomic_flag val_cpu_;
    };
};


