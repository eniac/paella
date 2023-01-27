#!/usr/bin/python3

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

expected_jct = 316 * 8 # in ns

parser = argparse.ArgumentParser()
parser.add_argument('csvfile', type=str)
args = parser.parse_args()

df = pd.read_csv(args.csvfile, delimiter='\t')
print(df)

plt.plot(df.index, df.JCT, '.', label='jobs latency')
plt.plot(df.index, [expected_jct for i in range(df.shape[0])], label='ideal job latency')

plt.legend()
plt.xlabel('Job index')
plt.ylabel('Latency (us)')
fname = f"{(args.csvfile)}.pdf"
print(f'Storing plot in {fname}')
plt.savefig(fname)
