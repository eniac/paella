#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-o', '--output_path', dest='output_path');
    parser.add_argument('-i', '--input_path', dest='input_paths', action='append');
    parser.add_argument('-a', '--algo_name', dest='algo_names', action='append');
    parser.add_argument('-l', '--line', dest='lines', type=int, action='append');
    parser.add_argument('-n', '--name', dest='names', action='append');
    parser.add_argument('--xlim', dest='xlim', type=float);
    parser.add_argument('--ylim', dest='ylim', type=float);
    parser.add_argument('--xaxis', choices=['throughput', 'rate'], type=str);
    parser.add_argument('--yaxis', dest='yaxises', choices=['mean', 'p50', 'p90', 'p95', 'p99'], action='append', type=str);
    parser.add_argument('--subplotx', dest='subplotx', type=int);
    parser.add_argument('--subploty', dest='subploty', type=int);

    args = parser.parse_args()

    yaxis_name2offset = {'mean': 0, 'p50': 1, 'p90': 2, 'p95': 3, 'p99': 4}

    num_inputs = len(args.input_paths)

    data = [np.genfromtxt(path, delimiter=',') for path in args.input_paths]

    fig, subplots = plt.subplots(args.subplotx, args.subploty, sharex=True, sharey=True)

    if args.xaxis == 'throughput':
        x_axis = [15000. / x[:, 1] * 1000000. for x in data]
        fig.supxlabel('Throughput (req/s)')
    else:
        x_axis = [1000000. / x[:, 0] for x in data]
        fig.supxlabel('Sending rate (req/s)')

    fig.supylabel('Latency (us)')

    for line, name, subplot in zip(args.lines, args.names, subplots.flat):
        for i in range(num_inputs):
            for yaxis in args.yaxises:
                subplot.errorbar(x_axis[i], data[i][:, line * 7 + yaxis_name2offset[yaxis] + 2], data[i][:, line * 7 + 6 + 2], label=args.algo_names[i] + ' ' + yaxis, fmt='x-', linewidth=1, markersize=2, elinewidth=0)

        subplot.set_title(name)
        subplot.legend()
        subplot.label_outer()

    fig.tight_layout()

    if args.xlim is not None:
        plt.xlim(0, args.xlim)
    if args.ylim is not None:
        plt.ylim(0, args.ylim)

    plt.savefig(args.output_path)

