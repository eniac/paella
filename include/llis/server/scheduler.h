#pragma once

#ifdef LLIS_SCHED_FIFO
#include <llis/server/scheduler_fifo.h>
#else
#include <llis/server/scheduler_full.h>
#endif

namespace llis {
namespace server {

#ifdef LLIS_SCHED_FIFO
using Scheduler = SchedulerFifo;
#else
using Scheduler = SchedulerFull;
#endif

}
}
