#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

fmts = ['X-', 'o-', '^-', 's-', 'D-', 'v-', 'p-', '*-', 'H-']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-o', '--output_path', dest='output_path');
    parser.add_argument('-i', '--input_path', dest='input_paths', action='append');
    parser.add_argument('-a', '--algo_name', dest='algo_names', action='append');
    parser.add_argument('-l', '--line', dest='lines', type=int, action='append');
    parser.add_argument('-n', '--name', dest='names', action='append');
    parser.add_argument('--yaxis', dest='yaxises', choices=['Mean', 'p50', 'p90', 'p95', 'p99'], action='append', type=str);
    parser.add_argument('-x', '--xlim', dest='xlim', type=float);
    parser.add_argument('-y', '--ylim', dest='ylim', type=float);

    args = parser.parse_args()

    yaxis_name2offset = {'Mean': 0, 'p50': 1, 'p90': 2, 'p95': 3, 'p99': 4}

    num_inputs = len(args.input_paths)

    data = [np.genfromtxt(path, delimiter=',') for path in args.input_paths]

    #plt.rcParams.update({'font.size': 6})
    plt.figure(figsize=(6.4, 3.9552))

    for i in range(num_inputs):
        for line, name, fmt in zip(args.lines, args.names, fmts):
            for yaxis in args.yaxises:
                plt.plot(data[i][:, 0], data[i][:, line * 7 + yaxis_name2offset[yaxis] + 3] / 1000., fmt, label=name, linewidth=1, markersize=2)

    plt.gca().invert_xaxis()
    plt.gca().set_xlim(500, 0)
    #plt.xlim(500, 0)
    plt.xlabel('Less Fair  <-     Fairness Threshold     ->  More Fair')
    if len(args.yaxises) == 1:
        plt.ylabel(args.yaxises[0] + ' Latency (ms)')
    else:
        plt.ylabel('Latency (ms)')

    plt.legend()
    #if args.xlim is not None:
    #    plt.xlim(0, args.xlim)
    #if args.ylim is not None:
    #    plt.ylim(0, args.ylim)

    plt.savefig(args.output_path, bbox_inches='tight')

