#!/usr/bin/python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

result_files = [f for f in os.listdir(os.curdir) if os.path.isfile(f) and f.endswith('.csv')]

dfs = []
for f in result_files:
    df = pd.read_csv(f, delimiter='\t')
    p99 = df.JCT.quantile(q=.99, interpolation='nearest')
    rate = df.RATE.values[0]
    ndf = pd.DataFrame([[p99, rate]] , columns=['p99', 'rate'])
    dfs.append(ndf)

df = pd.concat(dfs)
print(df)
plt.plot(df.rate, df.p99, '.')
plt.xlabel('Sending rate')
plt.ylabel('p99 (ns)')
plt.savefig('plot.pdf')
