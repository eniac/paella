#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/ipc/unix_datagram_socket.h>
#include <llis/ipc/defs.h>

namespace llis {
namespace server {

class ClientConnection {
  public:
    ClientConnection(ClientId client_id) : client_id_(client_id) {}

    void use_s2c_channel(ipc::ShmChannelCpuWriter&& s2c_channel) {
        s2c_channel_ = std::move(s2c_channel);
    }

    void use_s2c_socket(ipc::UnixDatagramSocket&& sock) {
        s2c_socket_ = std::move(sock);
    }

    ipc::ShmChannelCpuWriter* get_s2c_channel() {
        return &s2c_channel_;
    }

    ipc::UnixDatagramSocket* get_s2c_socket() {
        return &s2c_socket_;
    }

    ClientId get_client_id() const {
        return client_id_;
    }

  private:
    ClientId client_id_;
    
    ipc::ShmChannelCpuWriter s2c_channel_;
    ipc::UnixDatagramSocket s2c_socket_;
};

}
}

