#pragma once

#include "job_ref.h"

#include <llis/ipc/shared_memory_channel.h>

#include <cstdint>
#include <string>

namespace llis {

class Client {
  public:
    Client(std::string server_name);

    JobRef register_job(std::string path);

  private:
    //ipc::SharedMemoryChannel c2s_channel_;
};

}

