#pragma once

#include <llis/utils/gpu.h>

#include <atomic>

template <typename T, bool for_gpu>
class ThreadfenceWrapper {};

template <typename T>
class ThreadfenceWrapper<T, false> {
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
class ThreadfenceWrapper<T, true> {
  public:
    CUDA_HOSTDEV inline T load() const {
#ifdef __CUDA_ARCH__
        T val = val_;
        __threadfence_system();
        return val;
#else
        std::atomic<T>* tmp = reinterpret_cast<std::atomic<T>*>(const_cast<T*>(&val_));
        return tmp->load(std::memory_order_acquire);
#endif
    }

    CUDA_HOSTDEV inline void store(T desired) {
#ifdef __CUDA_ARCH__
        __threadfence_system();
        val_ = desired;
#else
        std::atomic<T>* tmp = reinterpret_cast<std::atomic<T>*>(const_cast<T*>(&val_));
        tmp->store(desired, std::memory_order_release);
#endif
    }

  private:
    volatile T val_;
};

