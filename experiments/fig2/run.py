#!/usr/bin/python3

import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('exp_label', type=str)
parser.add_argument('mode', type=str)
args = parser.parse_args()

for i in range(1000, 10000, 1000):
    cmd_args = ['./fig2', args.mode, args.exp_label, str(1e9/i)] # interval in ns
    p = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while(1):
        line = p.stdout.readline()
        print(line.decode('ascii'))
        if not line:
            break
    p.wait()
