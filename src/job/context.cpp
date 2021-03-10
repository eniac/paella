#include <llis/ipc/shm_primitive_channel.h>
#include <llis/ipc/shm_channel.h>
#include <llis/job/context.h>

namespace llis {
namespace job {

Job* Context::current_job_;
ipc::Gpu2SchedChannel Context::gpu2sched_channel_;
#ifdef LLIS_MEASURE_BLOCK_TIME
ipc::Gpu2SchedChannel Context::gpu2sched_block_time_channel_;
#endif
ipc::ShmChannel Context::mem2sched_channel_;

}
}

