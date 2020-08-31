#include <atomic>
#include <llis/ipc/shm_channel.h>

#include <cstring>
#include <cassert>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace llis {
namespace ipc {

size_t ShmChannel::next_aligned_pos(size_t next_pos, size_t align) {
    return (next_pos + align - 1) & ~(align - 1);
}

ShmChannel::ShmChannel(std::string name, size_t size) {
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

    total_size_ = sizeof(size_t);

    size_t read_pos_pos = next_aligned_pos(total_size_, alignof(std::atomic<size_t>));
    total_size_ = read_pos_pos + sizeof(std::atomic<size_t>);

    size_t write_pos_pos = next_aligned_pos(total_size_, alignof(std::atomic<size_t>));
    total_size_ = write_pos_pos + sizeof(std::atomic<size_t>);

    size_t writer_lock_pos = next_aligned_pos(total_size_, alignof(std::atomic_flag));
    total_size_ = writer_lock_pos + sizeof(std::atomic_flag);

    size_t ring_buf_offset = total_size_;

    total_size_ += size_;

    if (is_create_) {
        ftruncate(fd_, total_size_);
    }

    shm_ = reinterpret_cast<char*>(mmap(nullptr, total_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    ring_buf_ = shm_ + ring_buf_offset;

    read_pos_ = reinterpret_cast<std::atomic<size_t>*>(shm_ + read_pos_pos);
    write_pos_ = reinterpret_cast<std::atomic<size_t>*>(shm_ + write_pos_pos);
    writer_lock_ = reinterpret_cast<std::atomic_flag*>(shm_ + writer_lock_pos);

    if (is_create_) {
        *reinterpret_cast<size_t*>(shm_) = size;
        read_pos_->store(0);
        write_pos_->store(0);
        writer_lock_->clear();
    }
}

ShmChannel::~ShmChannel() {
    munmap(shm_, total_size_);
    close(fd_);
    if (is_create_) {
        shm_unlink(name_with_prefix_.c_str());
    }
}

void ShmChannel::read(void* buf, size_t size) {
    size_t size_to_read = size;
    size_t size_read = 0;
    while (size_to_read > 0) {
        size_t write_pos, read_pos;
        while ((write_pos = write_pos_->load(std::memory_order_acquire)) == (read_pos = read_pos_->load(std::memory_order_acquire))) {}

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
        read_pos_->store(read_pos, std::memory_order_release);
    }
}

void ShmChannel::write(void* buf, size_t size) {
    size_t size_to_write = size;
    size_t size_written = 0;

    while (size_to_write > 0) {
        size_t write_pos, read_pos;
        while (((write_pos = write_pos_->load(std::memory_order_acquire)) + 1) % size_ == (read_pos = read_pos_->load(std::memory_order_acquire))) {}

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
        
        memcpy(ring_buf_ + write_pos, reinterpret_cast<char*>(buf) + size_written, size_writing);

        size_to_write -= size_writing;
        size_written += size_writing;
        
        write_pos += size_writing;
        assert(write_pos <= size_);
        if (write_pos == size_) {
            write_pos = 0;
        }
        write_pos_->store(write_pos, std::memory_order_release);
    }
}

void ShmChannel::acquire_writer_lock() {
    while (!writer_lock_->test_and_set(std::memory_order_acquire));
}

void ShmChannel::release_writer_lock() {
    writer_lock_->clear(std::memory_order_release);
}

}
}

