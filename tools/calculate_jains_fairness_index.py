#!/usr/bin/env python3

import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', dest='output_path');
    parser.add_argument('-b', '--input_baseline_path', dest='input_baseline_paths', action='append');
    parser.add_argument('-i', '--input_path', dest='input_path');

    args = parser.parse_args()

    data = np.genfromtxt(args.input_path, delimiter=',')
    baseline_data = [np.genfromtxt(path, delimiter=',') for path in args.input_baseline_paths]

    baseline_latencies = np.array([x[2] for x in baseline_data], dtype=float)

    slowdown_factors = data[:, 9::7] / baseline_latencies

    fairness_indices = np.sum(slowdown_factors, axis=1, keepdims=True) ** 2. / (np.sum(slowdown_factors ** 2, axis=1, keepdims=True) * len(baseline_latencies))

    results = np.concatenate([data[:, 0:1], fairness_indices, slowdown_factors], axis=1)
    
    np.savetxt(args.output_path, results, delimiter=',')
