#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/utils/gpu.h>
#include <llis/utils/align.h>

#include <cstring>
#include <cassert>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace llis {
namespace ipc {

template <>
CUDA_HOSTDEV void* ShmChannelBase<false>::memcpy(void* dest, const void* src, size_t count) {
    return ::memcpy(dest, src, count);
}

template <>
CUDA_HOSTDEV void* ShmChannelBase<true>::memcpy(void* dest_, const void* src_, size_t count) {
    volatile char* dest = reinterpret_cast<volatile char*>(dest_);
    volatile const char* src = reinterpret_cast<volatile const char*>(src_);

    for (size_t i = 0; i < count; ++i) {
        dest[i] = src[i];
    }

    return dest_;
}

template <bool for_gpu>
ShmChannelBase<for_gpu>::ShmChannelBase(std::string name, size_t size) {
    connect(name, size);
}

template <bool for_gpu>
ShmChannelBase<for_gpu>::~ShmChannelBase() {
    disconnect();
}

template <bool for_gpu>
ShmChannelBase<for_gpu>::ShmChannelBase(ShmChannelBase<for_gpu>&& rhs) {
    *this = std::move(rhs);
}

template <bool for_gpu>
ShmChannelBase<for_gpu>& ShmChannelBase<for_gpu>::operator=(ShmChannelBase<for_gpu>&& rhs) {
    fd_ = rhs.fd_;
    shm_ = rhs.shm_;
    ring_buf_ = rhs.ring_buf_;
    size_ = rhs.size_;
    total_size_ = rhs.total_size_;
    is_create_ = rhs.is_create_;
    name_with_prefix_ = rhs.name_with_prefix_;
    read_pos_ = rhs.read_pos_;
    write_pos_ = rhs.write_pos_;
    writer_lock_ = rhs.writer_lock_;

    rhs.shm_ = nullptr;

    return *this;
}

template <bool for_gpu>
void ShmChannelBase<for_gpu>::connect(std::string name, size_t size) {
    shm_ = nullptr;

    if (name != "") {
        is_create_ = (size > 0);
        name_with_prefix_ = "llis:" + name;
        if (is_create_) {
            fd_ = shm_open(name_with_prefix_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        } else {
            fd_ = shm_open(name_with_prefix_.c_str(), O_RDWR, 0600);
        }
        // TODO: error handling

        if (is_create_) {
            size_ = size;
        } else {
            size_t* size_shm_ = reinterpret_cast<size_t*>(mmap(nullptr, sizeof(size_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
            size_ = *size_shm_;
            munmap(size_shm_, sizeof(size_t));
        }
    } else {
        is_create_ = true;
        name_with_prefix_ = "";
        size_ = size;
    }

    total_size_ = sizeof(size_t);

    size_t read_pos_pos = utils::next_aligned_pos(total_size_, alignof(AtomicWrapper<size_t, for_gpu>));
    total_size_ = read_pos_pos + sizeof(std::atomic<size_t>);

    size_t write_pos_pos = utils::next_aligned_pos(total_size_, alignof(AtomicWrapper<size_t, for_gpu>));
    total_size_ = write_pos_pos + sizeof(std::atomic<size_t>);

    size_t writer_lock_pos = utils::next_aligned_pos(total_size_, alignof(std::atomic_flag));
    total_size_ = writer_lock_pos + sizeof(std::atomic_flag);

    size_t ring_buf_offset = total_size_;

    total_size_ += size_;

    if (name_with_prefix_ != "") {
        if (is_create_) {
            ftruncate(fd_, total_size_);
        }
        shm_ = reinterpret_cast<char*>(mmap(nullptr, total_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    } else {
        shm_ = new char[total_size_];
    }

    if constexpr (for_gpu) {
        cudaHostRegister(shm_, total_size_, cudaHostRegisterDefault);
    }

    ring_buf_ = shm_ + ring_buf_offset;

    read_pos_ = reinterpret_cast<AtomicWrapper<size_t, for_gpu>*>(shm_ + read_pos_pos);
    write_pos_ = reinterpret_cast<AtomicWrapper<size_t, for_gpu>*>(shm_ + write_pos_pos);
    writer_lock_ = reinterpret_cast<AtomicLock<for_gpu>*>(shm_ + writer_lock_pos);

    if (is_create_) {
        *reinterpret_cast<size_t*>(shm_) = size;
        read_pos_->store(0);
        write_pos_->store(0);
        writer_lock_->init();
    }
}

template <bool for_gpu>
void ShmChannelBase<for_gpu>::connect(ShmChannelBase<for_gpu>* channel) {
    fd_ = -1;
    shm_ = channel->shm_;
    ring_buf_ = channel->ring_buf_;
    size_ = channel->size_;
    total_size_ = channel->total_size_;
    is_create_ = false;
    name_with_prefix_ = channel->name_with_prefix_;
    read_pos_ = channel->read_pos_;
    write_pos_ = channel->write_pos_;
    writer_lock_ = channel->writer_lock_;
}

template <bool for_gpu>
void ShmChannelBase<for_gpu>::disconnect() {
    if (is_connected()) {
        if (name_with_prefix_ != "") {
            munmap(shm_, total_size_);
            if (fd_ != -1) {
                close(fd_);
            }
            if (is_create_) {
                shm_unlink(name_with_prefix_.c_str());
            }
        } else {
            if (is_create_) {
                delete[] shm_;
            }
        }
        shm_ = nullptr;
    }
}

template <bool for_gpu>
bool ShmChannelBase<for_gpu>::is_connected() {
    return shm_ != nullptr;
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
        size_t size_reading = std::min(size_to_read, size_can_read);

        memcpy(reinterpret_cast<char*>(buf) + size_read, ring_buf_ + read_pos, size_reading);

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

        size_t size_writing = std::min(size_to_write, size_can_write);
        
        memcpy(ring_buf_ + write_pos, reinterpret_cast<const char*>(buf) + size_written, size_writing);

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

