#!/usr/bin/python3

import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('exp_label', type=str)
parser.add_argument('mode', type=str)
parser.add_argument('--num_hwq', type=str, default=32)
parser.add_argument('--iterate-hwq', action='store_true', default=False)
args = parser.parse_args()

def run_over_load(n_hwq: int):
    for i in range(100, 2000, 200):
        label = f'{args.exp_label}-{n_hwq}hwq'
        cmd_args = ['./fig2', args.mode, label, str(1e9/i)] # interval in ns
        p = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_DEVICE_MAX_CONNECTIONS=str(n_hwq)))
        while(1):
            line = p.stdout.readline()
            print(line.decode('ascii'))
            if not line:
                break
        p.wait()

if (args.iterate_hwq):
    n_hwq = 1
    while n_hwq <= 32:
        run_over_load(n_hwq)
        n_hwq *= 2
else:
    run_over_load(args.num_hwq)
