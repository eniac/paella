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
ShmPrimitiveChannelBase<T, for_gpu>::ShmPrimitiveChannelBase(std::string name, size_t count) {
    connect(name, count);
}

template <typename T, bool for_gpu>
ShmPrimitiveChannelBase<T, for_gpu>::~ShmPrimitiveChannelBase() {
    disconnect();
}

template <typename T, bool for_gpu>
ShmPrimitiveChannelBase<T, for_gpu>::ShmPrimitiveChannelBase(ShmPrimitiveChannelBase<T, for_gpu>&& rhs) {
    *this = std::move(rhs);
}

template <typename T, bool for_gpu>
ShmPrimitiveChannelBase<T, for_gpu>& ShmPrimitiveChannelBase<T, for_gpu>::operator=(ShmPrimitiveChannelBase<T, for_gpu>&& rhs) {
    fd_ = rhs.fd_;
    shm_ = rhs.shm_;
    ring_buf_ = rhs.ring_buf_;
    count_ = rhs.count_;
    total_size_ = rhs.total_size_;
    is_create_ = rhs.is_create_;
    name_with_prefix_ = rhs.name_with_prefix_;
    read_pos_ = rhs.read_pos_;
    write_pos_ = rhs.write_pos_;

    rhs.shm_ = nullptr;

    return *this;
}

template <typename T, bool for_gpu>
void ShmPrimitiveChannelBase<T, for_gpu>::connect(std::string name, size_t count) {
    shm_ = nullptr;

    if (name != "") {
        if constexpr (for_gpu) {
            fprintf(stderr, "GPU queue cannot be on shared memory\n");
        }

        is_create_ = (count > 0);
        name_with_prefix_ = "llis:pchannel:" + name;
        if (is_create_) {
            fd_ = shm_open(name_with_prefix_.c_str(), O_CREAT | O_RDWR, 0600);
        } else {
            fd_ = shm_open(name_with_prefix_.c_str(), O_RDWR, 0600);
        }
        // TODO: error handling

        if (is_create_) {
            count_ = count;
        } else {
            size_t* count_shm_ = reinterpret_cast<size_t*>(mmap(nullptr, sizeof(size_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
            count_ = *count_shm_;
            munmap(count_shm_, sizeof(size_t));
        }
    } else {
        is_create_ = true;
        name_with_prefix_ = "";
        if (count <= 0) {
            fprintf(stderr, "Count must be > 0\n");
        }
        count_ = count;
    }

    size_t size = count_ * sizeof(T);

    total_size_ = sizeof(size_t);

    size_t write_pos_pos = utils::next_aligned_pos(total_size_, alignof(AtomicWrapper<unsigned, for_gpu>));
    if constexpr (!for_gpu) {
        total_size_ = write_pos_pos + sizeof(AtomicWrapper<unsigned, for_gpu>);
    }

    size_t ring_buf_offset = utils::next_aligned_pos(total_size_, alignof(T));
    total_size_ = ring_buf_offset + size;

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

    ring_buf_ = reinterpret_cast<T*>(shm_ + ring_buf_offset);

    read_pos_ = 0;
    if constexpr (for_gpu) {
        cudaMalloc(&write_pos_, sizeof(AtomicWrapper<unsigned, for_gpu>));
    } else {
        write_pos_ = reinterpret_cast<AtomicWrapper<unsigned, for_gpu>*>(shm_ + write_pos_pos);
    }

    if (is_create_) {
        *reinterpret_cast<size_t*>(shm_) = count_;
        if constexpr (for_gpu) {
            cudaMemset(write_pos_, 0, sizeof(*write_pos_));
        } else {
            write_pos_->store(0);
        }

        memset(ring_buf_, 0, size);
    }
}

template <typename T, bool for_gpu>
void ShmPrimitiveChannelBase<T, for_gpu>::connect(ShmPrimitiveChannelBase<T, for_gpu>* channel) {
    fd_ = -1;
    shm_ = channel->shm_;
    ring_buf_ = channel->ring_buf_;
    count_ = channel->count_;
    total_size_ = channel->total_size_;
    is_create_ = false;
    name_with_prefix_ = channel->name_with_prefix_;
    read_pos_ = channel->read_pos_;
    write_pos_ = channel->write_pos_;
}

template <typename T, bool for_gpu>
void ShmPrimitiveChannelBase<T, for_gpu>::disconnect() {
    if (is_connected()) {
        if (name_with_prefix_ != "") {
            munmap(shm_, total_size_);
            if (fd_ != -1) {
                close(fd_);
            }
            if (is_create_) {
                shm_unlink(name_with_prefix_.c_str());
                if constexpr (for_gpu) {
                    cudaFree(write_pos_);
                }
            }
        } else {
            if (is_create_) {
                delete[] shm_;
            }
        }
        shm_ = nullptr;
    }
}

template <typename T, bool for_gpu>
bool ShmPrimitiveChannelBase<T, for_gpu>::is_connected() {
    return shm_ != nullptr;
}

template class ShmPrimitiveChannelBase<uint8_t, true>;
template class ShmPrimitiveChannelBase<uint16_t, true>;
template class ShmPrimitiveChannelBase<uint32_t, true>;
template class ShmPrimitiveChannelBase<uint64_t, true>;
#ifndef __CUDA_ARCH__
template class ShmPrimitiveChannelBase<uint8_t, false>;
template class ShmPrimitiveChannelBase<uint16_t, false>;
template class ShmPrimitiveChannelBase<uint32_t, false>;
template class ShmPrimitiveChannelBase<uint64_t, false>;
#endif

}
}

