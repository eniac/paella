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

    void use_s2c_socket(int sock) {
        s2c_socket_ = sock;
    }

    ipc::ShmChannel* get_s2c_channel() {
        return &s2c_channel_;
    }

    int get_s2c_socket() {
        return s2c_socket_;
    }

    ClientId get_client_id() const {
        return client_id_;
    }

  private:
    ClientId client_id_;
    
    ipc::ShmChannel s2c_channel_;
    int s2c_socket_;
};

}
}

