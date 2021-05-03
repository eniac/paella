#pragma once

#ifdef LLIS_SCHED_FIFO
#include <llis/server/scheduler_fifo.h>
#elif defined(LLIS_SCHED_FIFO2)
#include <llis/server/scheduler_fifo2.h>
#elif defined(LLIS_SCHED_FULL2)
#include <llis/server/scheduler_full2.h>
#else
#include <llis/server/scheduler_full.h>
#endif

namespace llis {
namespace server {

#ifdef LLIS_SCHED_FIFO
using Scheduler = SchedulerFifo;
#elif defined(LLIS_SCHED_FIFO2)
using Scheduler = SchedulerFifo2;
#elif defined(LLIS_SCHED_FULL2)
using Scheduler = SchedulerFull2;
#else
using Scheduler = SchedulerFull;
#endif

}
}
