#pragma once

#include <llis/ipc/atomic_wrapper.h>
#include <llis/gpu_utils.h>

#include <string>

namespace llis {
namespace ipc {

template <bool for_gpu>
class ShmChannelBase {
  public:
    ShmChannelBase() : shm_(nullptr) {}
    ShmChannelBase(std::string name, size_t size = 0);
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
 
    CUDA_HOSTDEV void read(void* buf, size_t size);
    CUDA_HOSTDEV void write(const void* buf, size_t size);

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
    CUDA_HOSTDEV void write(const T* buf) {
        write(buf, sizeof(T));
    }

    template <typename T>
    CUDA_HOSTDEV void write(T buf) {
        write(&buf, sizeof(T));
    }

    void write(const std::string& str) {
        write(str.size());
        write(str.c_str(), str.size());
    }

    void acquire_writer_lock();
    void release_writer_lock();

  private:
    CUDA_HOSTDEV static size_t next_aligned_pos(size_t next_pos, size_t align);

    int fd_;
    char* shm_;
    char* ring_buf_;
    size_t size_;
    size_t total_size_;
    bool is_create_;
    std::string name_with_prefix_;

    AtomicWrapper<size_t, for_gpu>* read_pos_;
    AtomicWrapper<size_t, for_gpu>* write_pos_;
    std::atomic_flag* writer_lock_;
};

using ShmChannel = ShmChannelBase<false>;
using ShmChannelGpu = ShmChannelBase<true>;

}
}

