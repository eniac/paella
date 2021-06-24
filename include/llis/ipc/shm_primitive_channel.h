#pragma once

#include <llis/utils/gpu.h>
#include <llis/ipc/atomic_wrapper.h>

#include <string>

namespace llis {
namespace ipc {

template <typename T, bool for_gpu>
class ShmPrimitiveChannelBase {
  public:
    ShmPrimitiveChannelBase() : shm_(nullptr) {}
    ShmPrimitiveChannelBase(std::string name, size_t count = 0);
    ShmPrimitiveChannelBase(ShmPrimitiveChannelBase<T, for_gpu>* channel) {
        connect(channel);
    }
    ShmPrimitiveChannelBase(size_t count) : ShmPrimitiveChannelBase("", count) {}
    ~ShmPrimitiveChannelBase();

    ShmPrimitiveChannelBase(const ShmPrimitiveChannelBase&) = delete;
    ShmPrimitiveChannelBase<T, for_gpu>& operator=(const ShmPrimitiveChannelBase<T, for_gpu>&) = delete;

    ShmPrimitiveChannelBase(ShmPrimitiveChannelBase&&);
    ShmPrimitiveChannelBase<T, for_gpu>& operator=(ShmPrimitiveChannelBase<T, for_gpu>&&);

    void connect(std::string name, size_t count = 0);
    void connect(ShmPrimitiveChannelBase<T, for_gpu>* channel);

    ShmPrimitiveChannelBase<T, for_gpu> fork() {
        ShmPrimitiveChannelBase<T, for_gpu> res;
        res.connect(this);
        return res;
    }

    void disconnect();
    bool is_connected();

    template <typename U>
    CUDA_HOSTDEV U read();
    template <typename U>
    CUDA_HOSTDEV void write(U val);

    template <typename U>
    CUDA_HOSTDEV bool can_read();

  private:
    int fd_;
    char* shm_;
    T* ring_buf_;
    size_t count_;
    size_t total_size_;
    bool is_create_;
    std::string name_with_prefix_;

    unsigned read_pos_;
    AtomicWrapper<unsigned, for_gpu>* write_pos_;

    T cached_head_;
};

template <typename T>
using ShmPrimitiveChannel = ShmPrimitiveChannelBase<T, false>;
template <typename T>
using ShmPrimitiveChannelGpu = ShmPrimitiveChannelBase<T, true>;

using Gpu2SchedChannel = ShmPrimitiveChannelGpu<uint64_t>;

}
}

#include "shm_primitive_channel_impl.h"

