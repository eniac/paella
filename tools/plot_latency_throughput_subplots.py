#!/usr/bin/env python3

from parse_input_kelvin import *
from parse_triton import *
from parse_clockwork import *

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import itertools

fmts = ['X-', 'o-', '^-', 's-', 'D-', 'v-', 'p-', '*-', 'H-']

yaxis_name2offset = {'Mean': 0, 'p50': 1, 'p90': 2, 'p95': 3, 'p99': 4}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output_path', dest='output_path');
    parser.add_argument('-i', '--input_path', dest='input_paths', action='append');
    parser.add_argument('-a', '--algo_name', dest='algo_names', action='append');
    parser.add_argument('-m', '--model_id', dest='model_ids', type=int, action='append');
    parser.add_argument('-n', '--model_name', dest='model_names', action='append');
    parser.add_argument('--xlim', dest='xlim', type=float);
    parser.add_argument('--ylim', dest='ylim', type=float);
    parser.add_argument('--xaxis', choices=['throughput', 'rate'], type=str);
    parser.add_argument('--yaxis', dest='yaxis_names', choices=['Mean', 'p50', 'p90', 'p95', 'p99'], action='append', type=str);
    parser.add_argument('--subplotx', dest='subplotx', type=int);
    parser.add_argument('--subploty', dest='subploty', type=int);
    parser.add_argument('--aspect', dest='aspect', type=float, default=0);

    args = parser.parse_args()

    data = []
    for path, algo_name in zip(args.input_paths, args.algo_names):
        print(algo_name)
        if algo_name == 'triton':
            data.append(parse_triton(path, args.xaxis, args.yaxis_names))
        elif algo_name == 'clockwork':
            data.append(parse_clockwork(path, args.xaxis, args.yaxis_names))
        else:
            data.append(parse_input_kelvin(path, args.xaxis, args.yaxis_names, args.model_ids))

    plt.rcParams.update({'font.size': 6})

    fig, subplots = plt.subplots(args.subplotx, args.subploty, sharex=True, sharey=True, layout='constrained')

    if args.subplotx * args.subploty > 1:
        subplots_flat = subplots.flat
    else:
        subplots_flat = [subplots]

    if args.xaxis == 'throughput':
        fig.supxlabel('Average Throughput (req/s)')
    else:
        fig.supxlabel('Sending rate (req/s)')

    if len(args.yaxis_names) == 1:
        fig.supylabel(args.yaxis_names[0] + ' Latency (ms)')
    else:
        fig.supylabel('Latency (ms)')

    for model_id, (model_name, subplot) in enumerate(zip(args.model_names, subplots_flat)):
        for data_per_algo, algo_name, fmt in zip(data, args.algo_names, itertools.cycle(fmts)):
            x_data_per_algo, y_data_per_algo = data_per_algo
            y_data_per_model = y_data_per_algo[model_id]

            for y_data_series, yaxis_name in zip(y_data_per_model, args.yaxis_names):
                if len(args.yaxis_names) == 1:
                    label = algo_name
                else:
                    label = algo_name + ' ' + yaxis_name

                subplot.plot(x_data_per_algo, y_data_series, fmt, label=label, linewidth=1, markersize=2)

        subplot.set_title(model_name)
        subplot.label_outer()
        if args.aspect != 0:
            subplot.set_box_aspect(args.aspect)

    # re-draw x ticks to be idempotent to experiment results
    x_axis = []
    for (x_axis_ticks, _) in data:
        if len(x_axis_ticks) > len(x_axis): #and max(x_axis_ticks) > max(x_axis):
            x_axis = x_axis_ticks
    for subplot in subplots_flat:
        subplot.set_xticks(x_axis)

    subplots_flat[0].legend()
    #fig.tight_layout()

    if args.xlim is not None:
        plt.xlim(0, args.xlim)
    if args.ylim is not None:
        plt.ylim(0, args.ylim)

    print(f'Saving figure at {args.output_path}')
    plt.savefig(args.output_path, bbox_inches='tight')

