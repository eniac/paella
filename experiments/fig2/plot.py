#!/usr/bin/python3

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot(args):
    plt.close('all')

    if not isinstance(args, argparse.Namespace):
        args = argparse.Namespace(**args)

    for exp_label in args.exp_labels:
        if args.verbose:
            print(f'Plotting data for {exp_label}')
        re_string = f"^{exp_label}-\d+-results.csv$"
        valid = re.compile(re_string)
        result_files = [f for f in os.listdir(os.curdir) if os.path.isfile(f) and re.match(valid, f)]

        dfs = []
        for f in result_files:
            df = pd.read_csv(f, delimiter='\t')
            n = int(df.shape[0] * .1)
            df.drop(index=df.index[:n], axis=0, inplace=True)
            p99 = df.JCT.quantile(q=.99, interpolation='nearest')
            median = df.JCT.quantile(q=.50, interpolation='nearest')
            rate = df.RATE.values[0]
            ndf = pd.DataFrame([[p99, median, rate]] , columns=['p99', 'median', 'rate'])
            dfs.append(ndf)

        df = pd.concat(dfs).sort_values('rate').reset_index(drop=True)
        if args.max_plot_rate > 0:
            df = df[df.rate <= args.max_plot_rate]

        if args.verbose:
            print(df)

        plt.plot(df.rate, df.p99, marker='x', label=exp_label)
        '''
        for index in range(df.shape[0]):
            x = df.loc[index].rate
            y = df.loc[index].p99
            print(x,y)
            plt.text(x, y, x, size=12)
        '''

    if args.ideal_jct > 0:
        plt.plot([r for r in df.rate.values], [args.ideal_jct for i in range(df.shape[0])], label=f'ideal job latency ({args.ideal_jct} us)')

    plt.legend()
    plt.xlabel('Sending rate')
    plt.ylabel('p99 (us)')
    plt.xticks(df.rate.values, df.rate.values)#, rotation=45)

    fname = f"{'-'.join(args.exp_labels)}.pdf"
    if args.verbose:
        print(f'Storing plot in {fname}')
    plt.savefig(fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_labels', nargs='+', type=str)
    parser.add_argument('-i', '--ideal-jct', type=int, help='Ideal JCT in us', default=0)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-m', '--max-plot-rate', type=int, default=0)
    args = parser.parse_args()

    plot(args)
