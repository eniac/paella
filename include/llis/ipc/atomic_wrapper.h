#pragma once

#include <llis/utils/gpu.h>

#include <atomic>

template <typename T, bool for_gpu>
class AtomicWrapper {};

template <typename T>
class AtomicWrapper<T, false> {
  public:
    inline T load() const {
        return val_.load(std::memory_order_acquire);
    }

    inline void store(T desired) {
        val_.store(desired, std::memory_order_release);
    }
    
  private:
    std::atomic<T> val_;
};

template <typename T>
class AtomicWrapper<T, true> {
  public:
    CUDA_HOSTDEV inline T load() const {
        T val = val_;
#ifdef __CUDA_ARCH__
        __threadfence_system();
#else
        atomic_thread_fence(std::memory_order_acquire);
#endif
        return val;
    }

    CUDA_HOSTDEV inline void store(T desired) {
#ifdef __CUDA_ARCH__
        __threadfence_system();
#else
        // TODO: check if this is good between CPU and GPU
        atomic_thread_fence(std::memory_order_release);
#endif
        val_ = desired;
    }

  private:
    volatile T val_;
};

