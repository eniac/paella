#pragma once

#include <cstdint>

namespace llis {

using ClientId = uint32_t;
using JobRefId = uint32_t;

enum class MsgType : uint32_t {
    REGISTER_CLIENT,
    REGISTER_JOB,
    LAUNCH_JOB,
    GROW_POOL
};

}

