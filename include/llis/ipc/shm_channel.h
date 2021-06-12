#pragma once

#include <llis/ipc/threadfence_wrapper.h>
#include <llis/ipc/atomic_lock.h>
#include <llis/utils/gpu.h>

#include <memory>
#include <string>

namespace llis {
namespace ipc {

template <bool for_gpu> class ShmChannelReader;
template <bool for_gpu> class ShmChannelWriter;


template <bool for_gpu>
class ShmChannelBase {
  public:
    ShmChannelBase() : shm_(nullptr) {}
    ShmChannelBase(const std::string& name, size_t size = 0);
    ShmChannelBase(ShmChannelBase<for_gpu>* channel) {
        connect(channel);
    }
    ShmChannelBase(size_t size) : ShmChannelBase("", size) {}
    ~ShmChannelBase();

    ShmChannelBase(const ShmChannelBase&) = delete;
    ShmChannelBase<for_gpu>& operator=(const ShmChannelBase<for_gpu>&) = delete;

    ShmChannelBase(ShmChannelBase&&);
    ShmChannelBase<for_gpu>& operator=(ShmChannelBase<for_gpu>&&);

    void connect(std::string name, size_t size = 0);
    void connect(ShmChannelBase<for_gpu>* channel);

    void disconnect();
    bool is_connected();

  protected:
    CUDA_HOSTDEV inline static void* my_memcpy(void* dest, const void* src, size_t count);

    int fd_;
    char* shm_;
    char* ring_buf_;
    size_t size_;
    size_t total_size_;
    bool is_create_;
    std::string name_with_prefix_;

    size_t cached_read_pos_;
    ThreadfenceWrapper<size_t, for_gpu>* read_pos_;
    size_t cached_write_pos_;
    ThreadfenceWrapper<size_t, for_gpu>* write_pos_;

    AtomicLock<for_gpu>* writer_lock_;
};

template <bool for_gpu>
class ShmChannelReader : public ShmChannelBase<for_gpu> {
  public:
    using ShmChannelBase<for_gpu>::ShmChannelBase;

    ShmChannelWriter<for_gpu> fork() {
        ShmChannelWriter<for_gpu> res;
        res.connect(this);
        return res;
    }

    CUDA_HOSTDEV void read(void* buf, size_t size);

    template <typename T>
    CUDA_HOSTDEV void read(T* buf) {
        read(buf, sizeof(T));
    }

    void read(std::string* str) {
        size_t len;
        read(&len);
        str->resize(len);
        read(&((*str)[0]), len);
    }

    template <typename T>
    CUDA_HOSTDEV void read(std::unique_ptr<T>* ptr) {
        T* ptr_tmp;
        read(&ptr_tmp);
        ptr->reset(ptr_tmp);
    }

    CUDA_HOSTDEV bool can_read();
};

template <bool for_gpu>
class ShmChannelWriter : public ShmChannelBase<for_gpu> {
  public:
    using ShmChannelBase<for_gpu>::ShmChannelBase;

    ShmChannelReader<for_gpu> fork() {
        ShmChannelReader<for_gpu> res;
        res.connect(this);
        return res;
    }

    CUDA_HOSTDEV void write(const void* buf, size_t size);

    template <typename T>
    CUDA_HOSTDEV void write(T buf) {
        write(&buf, sizeof(T));
    }

    void write(const std::string& str) {
        write(str.size());
        write(str.c_str(), str.size());
    }

    template <typename T>
    void write(std::unique_ptr<T>&& ptr) {
        T* ptr_tmp = ptr.release();
        write(reinterpret_cast<uintptr_t>(ptr_tmp));
    }

    CUDA_HOSTDEV void acquire_writer_lock();
    CUDA_HOSTDEV void release_writer_lock();
};

using ShmChannelCpuReader = ShmChannelReader<false>;
using ShmChannelCpuWriter = ShmChannelWriter<false>;
using ShmChannelGpuReader = ShmChannelReader<true>;
using ShmChannelGpuWriter = ShmChannelWriter<true>;

}
}

#include "shm_channel_impl.h"

