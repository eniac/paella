#pragma once

#include "llis/ipc/shm_channel.h"
#include <llis/ipc/defs.h>

namespace llis {
namespace server {

class ClientConnection {
  public:
    ClientConnection(ClientId client_id) : client_id_(client_id) {}

    void use_s2c_channel(ipc::ShmChannel&& s2c_channel) {
        s2c_channel_ = std::move(s2c_channel);
    }

    ipc::ShmChannel* get_s2c_channel() {
        return &s2c_channel_;
    }

  private:
    ClientId client_id_;
    
    ipc::ShmChannel s2c_channel_;
};

}
}

