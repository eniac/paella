#pragma once

#include "job_ref.h"

#include <llis/ipc/shm_channel_1to1.h>

#include <cstdint>
#include <string>

namespace llis {

class Client {
  public:
    Client(std::string server_name);

    JobRef register_job(std::string path);

  private:
    //ipc::ShmChannel1to1 c2s_channel_;
};

}

