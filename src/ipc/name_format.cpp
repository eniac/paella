#include <llis/ipc/name_format.h>
#include <llis/ipc/defs.h>

namespace llis {
namespace ipc {

std::string s2c_socket_name(const std::string& server_name, ClientId client_id) {
    return "llis-socket-s2c-" + server_name + "-" + std::to_string(client_id);
}

std::string s2c_channel_name(const std::string& server_name, ClientId client_id) {
    return "s2c:" + server_name + ":" + std::to_string(client_id);
}

std::string c2s_channel_name(const std::string& server_name) {
    return "c2s:" + server_name;
}

}
}

