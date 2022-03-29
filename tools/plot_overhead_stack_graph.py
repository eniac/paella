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
    parser.add_argument('-i', '--input_path', dest='input_path');
    parser.add_argument('-l', '--line', dest='line', type=int);

    args = parser.parse_args()

    data = np.genfromtxt(args.input_path, delimiter=' ', skip_header=args.line, max_rows=1, dtype=float) # Events of a particular job

    data = data.reshape((-1, 2)) # Each row is an event of a particular job

    prev_e = None
    prev_t = None

    queueing_delay = -1
    scheduling_overhead = 0
    kernel_launch_overhead = 0
    kernel_runtime = 0

    for i, (e, t) in enumerate(data):
        e = int(e)
        if prev_e == JobEvent.JOB_SUBMITTED and e == JobEvent.KERNEL_SCHED_START:
            queueing_delay = (t - prev_t)

        if prev_e == JobEvent.KERNEL_SCHED_START and (e == JobEvent.KERNEL_SCHED_ABORT or e == JobEvent.KERNEL_SUBMIT_START):
            scheduling_overhead += (t - prev_t)

        if prev_e == JobEvent.KERNEL_SUBMIT_START and e == JobEvent.KERNEL_SUBMIT_END:
            kernel_launch_overhead += (t - prev_t)

        if prev_e == JobEvent.KERNEL_SUBMIT_END and e == JobEvent.KERNEL_FINISHED:
            kernel_runtime += (t - prev_t)

        prev_e = e
        prev_t = t

    print('Queueing Delay:', queueing_delay)
    print('Scheduling Overhead:', scheduling_overhead)
    print('Kernel Launch Overhead:', kernel_launch_overhead)
    print('Kernel Runtime:', kernel_runtime)

    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(['Overheads'], [queueing_delay], width, label='Queueing Delay')
    ax.bar(['Overheads'], [scheduling_overhead], width, label='Scheduling Overhead')
    ax.bar(['Overheads'], [kernel_launch_overhead], width, label='Kernel Launch Overhead')
    ax.bar(['Overheads'], [kernel_runtime], width, label='Kernel Runtime')

    plt.legend()

    plt.savefig(args.output_path)

