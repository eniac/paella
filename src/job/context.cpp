#include <llis/job/context.h>

namespace llis {
namespace job {

Job* Context::current_job_;
ipc::ShmChannelGpu Context::gpu2sched_channel_;

}
}

