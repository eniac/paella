#pragma once

#include <llis/ipc/shm_channel.h>
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
CUDA_HOSTDEV void ShmChannelReader<for_gpu>::read(void* buf, size_t size) {
    size_t size_to_read = size;
    size_t size_read = 0;
    while (size_to_read > 0) {
        while ((this->cached_write_pos_ = this->write_pos_->load()) == this->cached_read_pos_) {}

        size_t final_write_pos;
        if (this->cached_write_pos_ < this->cached_read_pos_) {
            final_write_pos = this->size_;
        } else {
            final_write_pos = this->cached_write_pos_;
        }

        size_t size_can_read = final_write_pos - this->cached_read_pos_;
        size_t size_reading = ((size_to_read < size_can_read) ? size_to_read : size_can_read);

        this->my_memcpy(reinterpret_cast<char*>(buf) + size_read,
                        this->ring_buf_ + this->cached_read_pos_, size_reading);

        size_to_read -= size_reading;
        size_read += size_reading;

        this->cached_read_pos_ += size_reading;
        assert(this->cached_read_pos_ <= this->size_);
        if (this->cached_read_pos_ == this->size_) {
            this->cached_read_pos_ = 0;
        }
        this->read_pos_->store(this->cached_read_pos_);
    }
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelWriter<for_gpu>::write(const void* buf, size_t size) {
    size_t size_to_write = size;
    size_t size_written = 0;

    while (size_to_write > 0) {
        while ((this->cached_write_pos_ + 1) % this->size_ == (this->cached_read_pos_ = this->read_pos_->load())) {}

        size_t size_can_write;
        if (this->cached_read_pos_ <= this->cached_write_pos_) {
            if (this->cached_read_pos_ > 0) {
                size_can_write = this->size_ - this->cached_write_pos_;
            } else {
                size_can_write = this->size_ - this->cached_write_pos_ - 1;
            }
        } else {
            size_can_write = this->cached_read_pos_ - this->cached_write_pos_ - 1;
        }

        size_t size_writing = ((size_to_write < size_can_write) ? size_to_write : size_can_write);

        this->my_memcpy(this->ring_buf_ + this->cached_write_pos_,
                        reinterpret_cast<const char*>(buf) + size_written,
                        size_writing);

        size_to_write -= size_writing;
        size_written += size_writing;

        this->cached_write_pos_ += size_writing;
        assert(this->cached_write_pos_ <= this->size_);
        if (this->cached_write_pos_ == this->size_) {
            this->cached_write_pos_ = 0;
        }
        this->write_pos_->store(this->cached_write_pos_);
    }
}

template <bool for_gpu>
CUDA_HOSTDEV bool ShmChannelReader<for_gpu>::can_read() {
    if (this->cached_write_pos_ != this->cached_read_pos_) {
        return true;
    }
    return this->write_pos_->load() != this->cached_read_pos_;
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelWriter<for_gpu>::acquire_writer_lock() {
    this->writer_lock_->acquire();
}

template <bool for_gpu>
CUDA_HOSTDEV void ShmChannelWriter<for_gpu>::release_writer_lock() {
    this->writer_lock_->release();
}

}
}
