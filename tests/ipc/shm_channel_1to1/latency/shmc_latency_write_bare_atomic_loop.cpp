#define NUM_ITERS 1000000

#include <atomic>
#include <string>
#include <iostream>
#include <chrono>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    std::string name_with_prefix = "ml-on-apu:test";
    int fd_ = shm_open(name_with_prefix.c_str(), O_CREAT | O_RDWR, 0600);
    ftruncate(fd_, sizeof(std::atomic_bool));
    std::atomic_int* shm_ = reinterpret_cast<std::atomic_int*>(mmap(nullptr, sizeof(std::atomic_int), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));

    shm_->store(0, std::memory_order_release);
    shm_->store(2, std::memory_order_release);

    auto time1 = std::chrono::system_clock::now().time_since_epoch().count();

    for (int i = 1; i < NUM_ITERS; ++i) {
        shm_->store(i, std::memory_order_release);
        while (shm_->load(std::memory_order_acquire) != 0) {}
    }
    shm_->store(NUM_ITERS, std::memory_order_release);

    auto time2 = std::chrono::system_clock::now().time_since_epoch().count();

    std::cout << "time1: " << time1 << std::endl;
    std::cout << "time2: " << time2 << std::endl;
}
