#pragma once

#include <string>

#include <sys/un.h>

namespace llis {
namespace ipc {

class UnixDatagramSocket {
  public:
    UnixDatagramSocket();
    UnixDatagramSocket(const std::string& name);

    UnixDatagramSocket(UnixDatagramSocket&&);
    UnixDatagramSocket& operator=(UnixDatagramSocket&&);

    ~UnixDatagramSocket();

    void bind(const std::string& name);
    UnixDatagramSocket connect(const std::string& name);

    ssize_t write(const void* buf, size_t count);
    ssize_t read(void* buf, size_t count);

  private:
    UnixDatagramSocket(int socket);

    int socket_;
    bool is_owner_;

    sockaddr_un remote_addr_;
};

}
}

