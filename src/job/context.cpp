#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/context.h>

namespace llis {
namespace job {

Job* Context::current_job_;
ipc::Gpu2SchedChannel Context::gpu2sched_channel_;

}
}

