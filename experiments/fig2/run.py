#!/usr/bin/python3

import subprocess

for i in range(5000, 35000, 1000):
    args = ['./fig2', str(1e9/i)] # interval in ns
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while(1):
        line = p.stdout.readline()
        print(line.decode('ascii'))
        if not line:
            break
    p.wait()
