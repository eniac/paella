#pragma once

#include <string>
#include <atomic>

namespace llis {
namespace ipc {

class ShmChannel {
  public:
    ShmChannel() : fd_(0) {}
    ShmChannel(std::string name, size_t size = 0);
    ~ShmChannel();

    ShmChannel(const ShmChannel&) = delete;
    ShmChannel(ShmChannel&&) = default;
    ShmChannel& operator=(const ShmChannel&) = delete;
    ShmChannel& operator=(ShmChannel&&) = default;

    void connect(std::string name, size_t size = 0);
    void disconnect();
    bool is_connected();
 
    void read(void* buf, size_t size);
    void write(const void* buf, size_t size);

    template <typename T>
    void read(T* buf) {
        read(buf, sizeof(T));
    }

    void read(std::string* str) {
        size_t len;
        read(&len);
        str->resize(len);
        read(&str[0], len);
    }

    template <typename T>
    void write(const T* buf) {
        write(buf, sizeof(T));
    }

    template <typename T>
    void write(T buf) {
        write(&buf, sizeof(T));
    }

    void write(const std::string& str) {
        write(str.size());
        write(str.c_str(), str.size());
    }

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

