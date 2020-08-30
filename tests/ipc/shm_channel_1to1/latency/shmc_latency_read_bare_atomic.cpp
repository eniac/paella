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
    std::atomic_char* shm_ = reinterpret_cast<std::atomic_char*>(mmap(nullptr, sizeof(std::atomic_char), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));

    int a = shm_->load(std::memory_order_acquire);
    int b = shm_->load(std::memory_order_acquire);

    while (shm_->load(std::memory_order_acquire) != 3) {}

    //while (shm_->load(std::memory_order_acquire)) {}
    auto cur_time = std::chrono::system_clock::now().time_since_epoch().count();
    std::cout << "Current time since epoch: " << cur_time << std::endl;
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
}
