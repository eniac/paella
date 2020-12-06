#pragma once

#include <llis/utils/gpu.h>

#include <atomic>

template <typename T, bool for_gpu>
class AtomicWrapper {};

template <typename T>
class AtomicWrapper<T, false> {
  public:
    inline T load() const {
        return val_.load(std::memory_order_relaxed);
    }

    inline void store(T desired) {
        val_.store(desired, std::memory_order_relaxed);
    }

    inline void add(T val) {
        val_.fetch_add(val, std::memory_order_relaxed);
    }
    
  private:
    std::atomic<T> val_;
};

template <typename T>
class AtomicWrapper<T, true> {
  public:
    CUDA_HOSTDEV inline T load() const {
#ifdef __CUDA_ARCH__
        return val_;
#else
        static_assert(sizeof(std::atomic<T>) == sizeof(T));

        std::atomic<T>* tmp = reinterpret_cast<std::atomic<T>*>(const_cast<T*>(&val_));
        return tmp->load(std::memory_order_relaxed);
#endif
    }

    CUDA_HOSTDEV inline void store(T desired) {
#ifdef __CUDA_ARCH__
        val_ = desired;
#else
        static_assert(sizeof(std::atomic<T>) == sizeof(T));

        std::atomic<T>* tmp = reinterpret_cast<std::atomic<T>*>(const_cast<T*>(&val_));
        tmp->store(desired, std::memory_order_relaxed);
#endif
    }

    CUDA_HOSTDEV inline void add(T val) {
#ifdef __CUDA_ARCH__
        // TODO: _system is necessary if both CPU and GPU are writing, but not sure if it is necessary if only GPU is writing and CPU is reading
        atomicAdd(const_cast<T*>(&val_), val);
#else
        static_assert(sizeof(std::atomic<T>) == sizeof(T));

        std::atomic<T>* tmp = reinterpret_cast<std::atomic<T>*>(const_cast<T*>(&val_));
        tmp->fetch_add(val, std::memory_order_relaxed);
#endif
    }

    CUDA_HOSTDEV inline T inc(T compare) {
#ifdef __CUDA_ARCH__
        return atomicInc(const_cast<T*>(&val_), compare);
#else
        // TODO
#endif
    }

    CUDA_HOSTDEV inline T cas(T compare, T val) {
#ifdef __CUDA_ARCH__
        return atomicCAS(const_cast<T*>(&val_), compare, val);
#else
        // TODO
#endif
    }

  private:
    volatile T val_;
};

