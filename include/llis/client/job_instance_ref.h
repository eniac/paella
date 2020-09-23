#pragma once

#include "llis/ipc/shm_channel.h"
#include <llis/client/io_shm_entry.h>

namespace llis {
namespace client {

class JobRef;

class JobInstanceRef {
  public:
    JobInstanceRef(JobRef* job_ref, IoShmEntry io_shm_entry);

    void launch();
    void wait();

    void* get_input_ptr();
    void* get_output_ptr();

  private:
    JobRef* job_ref_;
    IoShmEntry io_shm_entry_;

    ipc::ShmChannel* s2c_channel_;
    ipc::ShmChannel* c2s_channel_;
};

}
}

