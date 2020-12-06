#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/utils/gpu.h>
#include <llis/utils/align.h>

#include <cuda_runtime_api.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <cassert>

namespace llis {
namespace ipc {

template <>
CUDA_HOSTDEV inline void* ShmChannelBase<false>::my_memcpy(void* dest, const void* src, size_t count) {
    return memcpy(dest, src, count);
}

template <>
CUDA_HOSTDEV inline void* ShmChannelBase<true>::my_memcpy(void* dest_, const void* src_, size_t count) {
    volatile char* dest = reinterpret_cast<volatile char*>(dest_);
    volatile const char* src = reinterpret_cast<volatile const char*>(src_);

    for (size_t i = 0; i < count; ++i) {
        dest[i] = src[i];
    }

    return dest_;
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelBase<for_gpu>::read(void* buf, size_t size) {
    size_t size_to_read = size;
    size_t size_read = 0;
    while (size_to_read > 0) {
        size_t write_pos, read_pos;
        while ((write_pos = write_pos_->load()) == (read_pos = read_pos_->load())) {}

        if (write_pos < read_pos) {
            write_pos = size_;
        }

        size_t size_can_read = write_pos - read_pos;
        size_t size_reading = ((size_to_read < size_can_read) ? size_to_read : size_can_read);

        my_memcpy(reinterpret_cast<char*>(buf) + size_read, ring_buf_ + read_pos, size_reading);

        size_to_read -= size_reading;
        size_read += size_reading;

        read_pos += size_reading;
        assert(read_pos <= size_);
        if (read_pos == size_) {
            read_pos = 0;
        }
        read_pos_->store(read_pos);
    }
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelBase<for_gpu>::write(const void* buf, size_t size) {
    size_t size_to_write = size;
    size_t size_written = 0;

    while (size_to_write > 0) {
        size_t write_pos, read_pos;
        while (((write_pos = write_pos_->load()) + 1) % size_ == (read_pos = read_pos_->load())) {}

        size_t size_can_write;
        if (read_pos <= write_pos) {
            if (read_pos > 0) {
                size_can_write = size_ - write_pos;
            } else {
                size_can_write = size_ - write_pos - 1;
            }
        } else {
            size_can_write = read_pos - write_pos - 1;
        }

        size_t size_writing = ((size_to_write < size_can_write) ? size_to_write : size_can_write);
        
        my_memcpy(ring_buf_ + write_pos, reinterpret_cast<const char*>(buf) + size_written, size_writing);

        size_to_write -= size_writing;
        size_written += size_writing;
        
        write_pos += size_writing;
        assert(write_pos <= size_);
        if (write_pos == size_) {
            write_pos = 0;
        }
        write_pos_->store(write_pos);
    }
}

template <bool for_gpu>
CUDA_HOSTDEV bool ShmChannelBase<for_gpu>::can_read() {
    return write_pos_->load() != read_pos_->load();
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelBase<for_gpu>::acquire_writer_lock() {
    writer_lock_->acquire();
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelBase<for_gpu>::release_writer_lock() {
    writer_lock_->release();
}

}
}

