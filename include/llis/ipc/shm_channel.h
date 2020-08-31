#pragma once

#include <string>
#include <atomic>

namespace llis {
namespace ipc {

class ShmChannel {
  public:
    ShmChannel(std::string name, size_t size = 0);
    ~ShmChannel();
 
    void read(void* buf, size_t size);
    void write(void* buf, size_t size);

    void acquire_writer_lock();
    void release_writer_lock();

  private:
    static size_t next_aligned_pos(size_t next_pos, size_t align);

    int fd_;
    char* shm_;
    char* ring_buf_;
    size_t size_;
    size_t total_size_;
    bool is_create_;
    std::string name_with_prefix_;

    std::atomic<size_t>* read_pos_;
    std::atomic<size_t>* write_pos_;
    std::atomic_flag* writer_lock_;
};

}
}

