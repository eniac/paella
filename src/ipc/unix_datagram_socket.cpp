#include <llis/ipc/unix_datagram_socket.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace llis {
namespace ipc {

UnixDatagramSocket::UnixDatagramSocket() {
    socket_ = socket(AF_UNIX, SOCK_DGRAM, 0);
    is_owner_ = true;
}

UnixDatagramSocket::UnixDatagramSocket(const std::string& name) {
    socket_ = socket(AF_UNIX, SOCK_DGRAM, 0);
    is_owner_ = true;

    bind(name);
}

UnixDatagramSocket::UnixDatagramSocket(int socket) : socket_(socket), is_owner_(false) {}

UnixDatagramSocket::UnixDatagramSocket(UnixDatagramSocket&& rhs) {
    *this = std::move(rhs);
}

UnixDatagramSocket& UnixDatagramSocket::operator=(UnixDatagramSocket&& rhs) {
    socket_ = rhs.socket_;
    is_owner_ = rhs.is_owner_;
    remote_addr_ = rhs.remote_addr_;

    rhs.socket_ = -1;
    rhs.is_owner_ = false;
    
    return *this;
}

UnixDatagramSocket::~UnixDatagramSocket() {
    if (is_owner_) {
        close(socket_);

        is_owner_ = false;
    }

    socket_ = -1;
}

void UnixDatagramSocket::bind(const std::string& name) {
    sockaddr_un addr;
    bzero(&addr, sizeof(addr));
    addr.sun_family = AF_UNIX;
    // TODO: check length. It should be < 108 bytes
    strncpy(addr.sun_path + 1, name.c_str(), 107);

    // TODO: check if it succeeds
    (void)::bind(socket_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
}

UnixDatagramSocket UnixDatagramSocket::connect(const std::string& name) {
    UnixDatagramSocket res(socket_);

    bzero(&res.remote_addr_, sizeof(res.remote_addr_));
    res.remote_addr_.sun_family = AF_UNIX;
    // TODO: check length. It should be < 108 bytes
    strncpy(res.remote_addr_.sun_path + 1, name.c_str(), 107);

    return res;
}

ssize_t UnixDatagramSocket::write(const void* buf, size_t count) {
    return sendto(socket_, buf, count, 0, reinterpret_cast<const sockaddr*>(&remote_addr_), sizeof(remote_addr_));
}

ssize_t UnixDatagramSocket::read(void* buf, size_t count) {
    return ::read(socket_, buf, count);
}

}
}

