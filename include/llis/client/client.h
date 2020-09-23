#pragma once

#include "job_ref.h"

#include <llis/ipc/shm_channel.h>
#include <llis/ipc/defs.h>

#include <cstdint>
#include <string>

namespace llis {
namespace client {

class Client {
  public:
    Client(std::string server_name);

    JobRef register_job(std::string path);

    ClientId get_client_id() const {
        return client_id_;
    }

    ipc::ShmChannel* get_c2s_channel() {
        return &c2s_channel_;
    }

    ipc::ShmChannel* get_s2c_channel() {
        return &s2c_channel_;
    }

  private:
    void generate_client_id();
    void create_s2c_channel();
    void reconnect_s2c_channel();
    void register_client();

    std::string server_name_;

    ClientId client_id_;

    ipc::ShmChannel c2s_channel_;
    ipc::ShmChannel s2c_channel_;
};

}
}

