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
    CUDA_PROFILER_START,
    CUDA_PROFILER_STOP,
    PROFILER_CMD,
    EXIT_CMD
};

enum class ProfilerMsgType : uint32_t {
    SET_RECORD_KERNEL_INFO,
    UNSET_RECORD_KERNEL_INFO,
    SET_RECORD_BLOCK_EXEC_TIME,
    UNSET_RECORD_BLOCK_EXEC_TIME,
    SET_RECORD_KERNEL_BLOCK_MIS_ALLOC,
    UNSET_RECORD_KERNEL_BLOCK_MIS_ALLOC,
    SET_RECORD_RUN_NEXT_TIMES,
    UNSET_RECORD_RUN_NEXT_TIMES,
    SET_RECORD_JOB_EVENTS,
    UNSET_RECORD_JOB_EVENTS,
    SET_RECORD_RESOURCE_EVENTS,
    UNSET_RECORD_RESOURCE_EVENTS,
    SAVE
};

}

