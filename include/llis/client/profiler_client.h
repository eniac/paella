#pragma once

#include <llis/ipc/shm_channel.h>

#include <string>

namespace llis {
namespace client {

class ProfilerClient {
  public:
    ProfilerClient(ipc::ShmChannelCpuWriter* c2s_channel) : c2s_channel_(c2s_channel) {}

    void set_record_kernel_info();
    void unset_record_kernel_info();

    void set_record_block_exec_time();
    void unset_record_block_exec_time();

    void set_record_kernel_block_mis_alloc();
    void unset_record_kernel_block_mis_alloc();

    void set_record_run_next_times();
    void unset_record_run_next_times();

    void save(const std::string& path);

  private:
    ipc::ShmChannelCpuWriter* c2s_channel_;
};

}
}

