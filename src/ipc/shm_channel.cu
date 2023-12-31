#include <llis/ipc/shm_channel.h>
#include <llis/utils/gpu.h>
#include <llis/utils/align.h>
#include <llis/utils/error.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <cassert>

namespace llis {
namespace ipc {

template <bool for_gpu>
ShmChannelBase<for_gpu>::ShmChannelBase(const std::string& name, size_t size) {
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
    this->fd_ = rhs.fd_;
    this->shm_ = rhs.shm_;
    this->ring_buf_ = rhs.ring_buf_;
    this->size_ = rhs.size_;
    this->total_size_ = rhs.total_size_;
    this->is_create_ = rhs.is_create_;
    this->name_with_prefix_ = rhs.name_with_prefix_;
    this->cached_read_pos_ = rhs.cached_read_pos_;
    this->read_pos_ = rhs.read_pos_;
    this->cached_write_pos_ = rhs.cached_write_pos_;
    this->write_pos_ = rhs.write_pos_;
    this->writer_lock_ = rhs.writer_lock_;

    rhs.shm_ = nullptr;

    return *this;
}

template <bool for_gpu>
void ShmChannelBase<for_gpu>::connect(std::string name, size_t size) {
    shm_ = nullptr;

    if (name != "") {
        is_create_ = (size > 0);
        name_with_prefix_ = "llis:channel:" + name;
        if (is_create_) {
            fd_ = shm_open(name_with_prefix_.c_str(), O_CREAT | O_RDWR, 0600);
        } else {
            fd_ = shm_open(name_with_prefix_.c_str(), O_RDWR, 0600);
        }
        utils::error_throw_posix(fd_);

        if (is_create_) {
            size_ = size;
        } else {
            size_t* size_shm_ = reinterpret_cast<size_t*>(mmap(nullptr, sizeof(size_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
            utils::error_throw_posix((uintptr_t)size_shm_, 0);
            size_ = *size_shm_;
            utils::warn_log_posix(munmap(size_shm_, sizeof(size_t)));
        }
    } else {
        is_create_ = true;
        name_with_prefix_ = "";
        size_ = size;
    }

    total_size_ = sizeof(size_t);

    size_t read_pos_pos = utils::next_aligned_pos(total_size_, alignof(ThreadfenceWrapper<size_t, for_gpu>));
    total_size_ = read_pos_pos + sizeof(ThreadfenceWrapper<size_t, for_gpu>);

    size_t write_pos_pos = utils::next_aligned_pos(total_size_, alignof(ThreadfenceWrapper<size_t, for_gpu>));
    total_size_ = write_pos_pos + sizeof(ThreadfenceWrapper<size_t, for_gpu>);

    size_t writer_lock_pos = utils::next_aligned_pos(total_size_, alignof(AtomicLock<for_gpu>));
    total_size_ = writer_lock_pos + sizeof(AtomicLock<for_gpu>);

    size_t ring_buf_offset = total_size_;

    total_size_ += size_;

    if (name_with_prefix_ != "") {
        if (is_create_) {
            utils::error_throw_posix(ftruncate(fd_, total_size_));
        }
        shm_ = reinterpret_cast<char*>(mmap(nullptr, total_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    } else {
        shm_ = reinterpret_cast<char*>(mmap(nullptr, total_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    }
    utils::error_throw_posix((uintptr_t)shm_, 0);

    if constexpr (for_gpu) {
        utils::error_throw_posix(mlock(shm_, total_size_));
        utils::error_throw_cuda(cudaHostRegister(shm_, total_size_, cudaHostRegisterDefault));
    }

    ring_buf_ = shm_ + ring_buf_offset;

    read_pos_ = reinterpret_cast<ThreadfenceWrapper<size_t, for_gpu>*>(shm_ + read_pos_pos);
    write_pos_ = reinterpret_cast<ThreadfenceWrapper<size_t, for_gpu>*>(shm_ + write_pos_pos);
    writer_lock_ = reinterpret_cast<AtomicLock<for_gpu>*>(shm_ + writer_lock_pos);

    if (is_create_) {
        *reinterpret_cast<size_t*>(shm_) = size;
        read_pos_->store(0);
        cached_read_pos_ = 0;
        write_pos_->store(0);
        cached_write_pos_ = 0;
        writer_lock_->init();
    } else {
        cached_read_pos_ = read_pos_->load();
        cached_write_pos_ = write_pos_->load();
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
    cached_read_pos_ = channel->cached_read_pos_;
    read_pos_ = channel->read_pos_;
    cached_write_pos_ = channel->cached_write_pos_;
    write_pos_ = channel->write_pos_;
    writer_lock_ = channel->writer_lock_;
}

template <bool for_gpu>
void ShmChannelBase<for_gpu>::disconnect() {
    if (is_connected()) {
        if (name_with_prefix_ != "") {
            utils::warn_log_posix(munmap(shm_, total_size_));
            if (fd_ != -1) {
                utils::warn_log_posix(close(fd_));
            }
            if (is_create_) {
                utils::warn_log_posix(shm_unlink(name_with_prefix_.c_str()));
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

template class ShmChannelBase<true>;
#ifndef __CUDA_ARCH__
template class ShmChannelBase<false>;
#endif

}
}

