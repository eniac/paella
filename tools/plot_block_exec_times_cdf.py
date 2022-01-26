#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    input_paths = sys.argv[2::2]
    labels = sys.argv[3::2]
    num_inputs = len(input_paths)

    data = [np.genfromtxt(path, delimiter=' ') for path in input_paths]

    for i in range(num_inputs):
        exec_times = data[i][:, 1] - data[i][:, 0]
        exec_times = np.sort(exec_times)
        plt.plot(exec_times, np.arange(len(exec_times)) / len(exec_times), '-', label=labels[i])

    plt.legend()
    #plt.ylim(0, 1000000)

    plt.savefig(output_path)


