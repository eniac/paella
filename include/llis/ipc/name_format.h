#pragma once

#include <llis/ipc/defs.h>

#include <string>

namespace llis {
namespace ipc {

std::string s2c_socket_name(const std::string& server_name, ClientId client_id);
std::string s2c_channel_name(const std::string& server_name, ClientId client_id);
std::string c2s_channel_name(const std::string& server_name);

}
}

