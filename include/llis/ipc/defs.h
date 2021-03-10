#pragma once

#include <cstdint>

namespace llis {

using ClientId = uint32_t;
using JobRefId = uint32_t;
using JobInstanceRefId = uint32_t;
using JobId = uint32_t;

enum class MsgType : uint32_t {
    REGISTER_CLIENT,
    REGISTER_JOB,
    LAUNCH_JOB,
    GROW_POOL,
    PROFILER_CMD
};

enum class ProfilerMsgType : uint32_t {
    SET_RECORD_KERNEL_EXEC_TIME,
    UNSET_RECORD_KERNEL_EXEC_TIME,
    SET_RECORD_BLOCK_EXEC_TIME,
    UNSET_RECORD_BLOCK_EXEC_TIME,
    SAVE
};

}

