#ifndef LLIS_TIME_H_IS_INCLUDED
#define LLIS_TIME_H_IS_INCLUDED

#define CPU_FREQ 2.5 // Adjust with your cpu speed

#include <llis/utils/ops.h>
#include <boost/chrono.hpp>
using hr_clock = boost::chrono::steady_clock;
typedef hr_clock::time_point tp;

uint64_t since_epoch(const tp &time);
uint64_t ns_diff(const tp &start, const tp &end);

static const auto system_start_time = hr_clock::now();

static inline uint64_t cycles_to_ns(uint64_t time) {
    return time / CPU_FREQ;
}

#endif /* LLIS_TIME_H_IS_INCLUDED */
