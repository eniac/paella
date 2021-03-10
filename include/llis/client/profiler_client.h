#pragma once

#include <llis/ipc/shm_channel.h>

#include <string>

namespace llis {
namespace client {

class ProfilerClient {
  public:
    ProfilerClient(ipc::ShmChannel* c2s_channel) : c2s_channel_(c2s_channel) {}

    void set_record_kernel_exec_time();
    void unset_record_kernel_exec_time();

    void set_record_block_exec_time();
    void unset_record_block_exec_time();

    void save(const std::string& path);

  private:
    ipc::ShmChannel* c2s_channel_;
};

}
}

