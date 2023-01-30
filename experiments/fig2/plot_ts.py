#!/usr/bin/python3

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

pd.set_option('display.max_rows', 500)

expected_jct = 316 * 8 # in us

parser = argparse.ArgumentParser()
parser.add_argument('csvfile', type=str)
parser.add_argument('-i', '--ideal-jct', type=int, help='Ideal JCT in us', default=expected_jct)
args = parser.parse_args()

df = pd.read_csv(args.csvfile, delimiter='\t')
print(df)

plt.plot(df.index, df.JCT, '.', label='jobs latency')

if args.ideal_jct > 0:
    plt.plot(df.index, [args.ideal_jct for i in range(df.shape[0])], label='ideal job latency')

plt.legend()
plt.xlabel('Job index')
plt.ylabel('Latency (us)')
fname = f'{args.csvfile}.pdf'
print(f'Storing plot in {fname}')
plt.savefig(fname)
