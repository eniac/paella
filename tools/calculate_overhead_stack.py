#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
from enum import IntEnum

class JobEvent(IntEnum):
    JOB_SUBMITTED = 0
    KERNEL_SCHED_START = 1
    KERNEL_SCHED_ABORT = 2
    KERNEL_SUBMIT_START = 3
    KERNEL_SUBMIT_END = 4
    KERNEL_FINISHED = 5
    JOB_FINISHED = 6

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-o', '--output_path', dest='output_path');
    parser.add_argument('-g', '--output_graph_path', dest='output_graph_path');
    parser.add_argument('-i', '--input_path', dest='input_path');
    parser.add_argument('-l', '--line', dest='line', type=int);

    args = parser.parse_args()

    data = np.genfromtxt(args.input_path, delimiter=' ', dtype=float) # Events of a particular job

    queueing_delay_sum = 0
    scheduling_overhead_sum = 0
    kernel_launch_overhead_sum = 0
    kernel_runtime_sum = 0
    kernel_interval_sum = 0

    for line in data:
        line = line.reshape((-1, 2)) # Each row is an event of a particular job

        prev_e = None
        prev_t = None
        prev_kernel_finish_time = -1

        queueing_delay = -1
        scheduling_overhead = 0
        kernel_launch_overhead = 0
        kernel_runtime = 0
        kernel_interval = 0

        for i, (e, t) in enumerate(line):
            e = int(e)
            if prev_e == JobEvent.JOB_SUBMITTED and e == JobEvent.KERNEL_SCHED_START:
                queueing_delay = (t - prev_t)

            if prev_e == JobEvent.KERNEL_SCHED_START and (e == JobEvent.KERNEL_SCHED_ABORT or e == JobEvent.KERNEL_SUBMIT_START):
                scheduling_overhead += (t - prev_t)

            if prev_e == JobEvent.KERNEL_SUBMIT_START and e == JobEvent.KERNEL_SUBMIT_END:
                kernel_launch_overhead += (t - prev_t)

            if prev_e == JobEvent.KERNEL_SUBMIT_END and e == JobEvent.KERNEL_FINISHED:
                kernel_runtime += (t - prev_t)

            if e == JobEvent.KERNEL_FINISHED:
                prev_kernel_finish_time = t

            if e == JobEvent.KERNEL_SUBMIT_START and prev_kernel_finish_time != -1:
                kernel_interval += (t - prev_kernel_finish_time)

            prev_e = e
            prev_t = t

        queueing_delay_sum += queueing_delay
        scheduling_overhead_sum += scheduling_overhead
        kernel_launch_overhead_sum += kernel_launch_overhead
        kernel_runtime_sum += kernel_runtime
        kernel_interval_sum += kernel_interval

    num_jobs = np.shape(data)[0]
    queueing_delay_mean = queueing_delay_sum / num_jobs
    scheduling_overhead_mean = scheduling_overhead_sum / num_jobs
    kernel_launch_overhead_mean = kernel_launch_overhead_sum / num_jobs
    kernel_runtime_mean = kernel_runtime_sum / num_jobs
    kernel_interval_mean = kernel_interval_sum / num_jobs

    with open(args.output_path, 'w') as f:
        f.write('Queueing Delay,Scheduling Overhead,Kernel Launch Overhead,Kernel Runtime,Kernel Interval\n')
        f.write('{},{},{},{},{}\n'.format(queueing_delay_mean, scheduling_overhead_mean, kernel_launch_overhead_mean, kernel_runtime_mean, kernel_interval_mean))

    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(['Overheads'], [queueing_delay], width, label='Queueing Delay')
    ax.bar(['Overheads'], [scheduling_overhead], width, label='Scheduling Overhead')
    ax.bar(['Overheads'], [kernel_launch_overhead], width, label='Kernel Launch Overhead')
    ax.bar(['Overheads'], [kernel_runtime], width, label='Kernel Runtime')
    ax.bar(['Overheads'], [kernel_interval], width, label='Kernel Interval')

    plt.legend()

    plt.savefig(args.output_graph_path)

