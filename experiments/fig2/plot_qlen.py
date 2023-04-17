#!/usr/bin/python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

expected_jct = 316 * 8 # in ns

parser = argparse.ArgumentParser()
parser.add_argument('exp_labels', nargs='+', type=str)
args = parser.parse_args()

for exp_label in args.exp_labels:
    print(f'Plotting data for {exp_label}')
    re_string = f"^{exp_label}-\d+-qlen-results.csv$"
    valid = re.compile(re_string)
    result_files = [f for f in os.listdir(os.curdir) if os.path.isfile(f) and re.match(valid, f)]
    print(result_files)
    for f in result_files:
        df = pd.read_csv(f, delimiter='\t')
        plt.plot(df.TIME, df.QLEN, label=f.split('qlen')[0].split('-')[-2])


plt.legend()
plt.xlabel('Sending time (seconds)')
plt.ylabel('Queue length')

fname = f"{'-'.join(args.exp_labels)}-qlen.pdf"
print(f'Storing plot in {fname}')
plt.savefig(fname)
