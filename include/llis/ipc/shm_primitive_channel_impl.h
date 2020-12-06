#pragma once

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/utils/gpu.h>
#include <llis/utils/align.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <cassert>

namespace llis {
namespace ipc {

template <typename T, bool for_gpu>
template <typename U>
CUDA_HOSTDEV U ShmPrimitiveChannelBase<T, for_gpu>::read() {
    static_assert(sizeof(T) == sizeof(U), "The type being read must be of the same size as the type of the channel");

    U* ptr = reinterpret_cast<U*>(ring_buf_ + read_pos_);

    while (!ptr->can_read());

    U res;
    *reinterpret_cast<T*>(&res) = reinterpret_cast<AtomicWrapper<T, for_gpu>*>(ptr)->load();

    ptr->set_can_write();

    if (read_pos_ == count_ - 1) {
        read_pos_ = 0;
    } else {
        ++read_pos_;
    }

    return res;
}

template <typename T, bool for_gpu>
template <typename U>
CUDA_HOSTDEV void ShmPrimitiveChannelBase<T, for_gpu>::write(U val) {
    // TODO: it is probably possible to remove the critical session between acquire and store
    // Not sure which one has better performance

    static_assert(sizeof(T) == sizeof(U), "The type being written must be of the same size as the type of the channel");

    size_t write_pos = write_pos_->inc(count_ - 1);
    U* ptr = reinterpret_cast<U*>(ring_buf_ + write_pos);

    reinterpret_cast<AtomicWrapper<T, for_gpu>*>(ptr)->store(*reinterpret_cast<T*>(&val));
}

template <typename T, bool for_gpu>
template <typename U>
CUDA_HOSTDEV bool ShmPrimitiveChannelBase<T, for_gpu>::can_read() {
    U* ptr = reinterpret_cast<U*>(ring_buf_ + read_pos_);

    return ptr->can_read();
}


}
}

