#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/finished_block_notifier.h>

#include <vector>

namespace llis {
namespace job {

FinishedBlockNotifier::FinishedBlockNotifier(ipc::Gpu2SchedChannel* gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
        , ipc::Gpu2SchedChannel* gpu2sched_block_time_channel
#endif
) {
    gpu2sched_channel_ = gpu2sched_channel->fork();
#ifdef LLIS_MEASURE_BLOCK_TIME
    gpu2sched_block_time_channel_ = gpu2sched_block_time_channel->fork();
#endif
}

FinishedBlockNotifier* FinishedBlockNotifier::create_array(unsigned num, ipc::Gpu2SchedChannel* gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
        , ipc::Gpu2SchedChannel* gpu2sched_block_time_channel
#endif
) {
    FinishedBlockNotifier* res;
    cudaMalloc((void**)&res, num * sizeof(FinishedBlockNotifier));

    std::vector<FinishedBlockNotifier> tmp;
    tmp.reserve(num);
    for (unsigned i = 0; i < num; ++i) {
        tmp.emplace_back(gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
                , gpu2sched_block_time_channel
#endif
            );
    }

    cudaMemcpy(res, tmp.data(), num * sizeof(FinishedBlockNotifier), cudaMemcpyHostToDevice);

    return res;
}

void FinishedBlockNotifier::free_array(FinishedBlockNotifier* ptr) {
    cudaFree(ptr);
}

}
}

