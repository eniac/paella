#pragma once

#include <llis/job/job.h>

namespace llis {
namespace job {

void memset_res(size_t count, Job* job);
void memset(void* ptr, int val, size_t count, Job* job, ipc::ShmChannelGpu* gpu2sched_channel);

}
}

