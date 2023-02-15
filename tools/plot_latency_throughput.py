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
    parser.add_argument('-x', '--xlim', dest='xlim', type=float);
    parser.add_argument('-y', '--ylim', dest='ylim', type=float);

    args = parser.parse_args()

    num_inputs = len(args.input_paths)

    data = [np.genfromtxt(path, delimiter=',') for path in args.input_paths]

    throughputs = [15000. / x[:, 1] * 1000000. for x in data]

    rates = [1000000. / x[:, 0] for x in data]

    for i in range(num_inputs):
        for line, name in zip(args.lines, args.names):
            plt.errorbar(throughputs[i], data[i][:, line * 7 + 2], data[i][:, line * 7 + 6 + 2], label=args.algo_names[i] + ' ' + name, fmt='x-', linewidth=1, markersize=2, elinewidth=0)

    plt.legend()
    if args.xlim is not None:
        plt.xlim(0, args.xlim)
    if args.ylim is not None:
        plt.ylim(0, args.ylim)

    plt.savefig(args.output_path)

