#pragma once

#include <llis/gpu_utils.h>

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
        return val_;
    }

    CUDA_HOSTDEV inline void store(T desired) {
        val_ = desired;
    }

  private:
    volatile T val_;
};

