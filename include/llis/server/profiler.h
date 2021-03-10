#pragma once

#include <llis/ipc/shm_channel.h>

#include <chrono>
#include <vector>

namespace llis {
namespace server {

class Profiler {
  public:
    Profiler(ipc::ShmChannel* c2s_channel) : c2s_channel_(c2s_channel) {}

    void handle_cmd();
    void handle_cmd_save();

    void save(const std::string& path);

    void record_kernel_exec_time(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time);
    void record_block_exec_time(unsigned long long start_time, unsigned long long end_time);
    //void record_kernel_sm_exec_time(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time);

  private:
    ipc::ShmChannel* c2s_channel_;

    bool kernel_exec_times_flag_ = false;
    std::vector<std::pair<std::chrono::time_point<std::chrono::steady_clock>, std::chrono::time_point<std::chrono::steady_clock>>> kernel_exec_times_;

    bool block_exec_times_flag_ = false;
    std::vector<std::pair<unsigned long long, unsigned long long>> block_exec_times_;
    //std::vector<std::pair<std::chrono::time_point<std::chrono::steady_clock>, std::chrono::time_point<std::chrono::steady_clock>>> block_exec_times_;
};

}
}

