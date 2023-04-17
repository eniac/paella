#!/usr/bin/python3

import pandas as pd
import numpy as np
import argparse
import sys
import os

'''
Triton result files are in the format results_[sending rate]_[sigma].csv, e.g., newmix3_sops23.yaml_100_1.5.csv
Columns are [ ID      MODEL   SEND    RECEIVE LATENCY ]
SEND/RECEIVE are in cycles (needs to be divided by 2195 (cycles/us) to get useconds)
LATENCY is in useconds

We return all times in milliseconds
'''

def get_all_mean(df):
    return df.LATENCY.mean() / 1e3

def get_models_mean(df):
    return df.groupby('MODEL').LATENCY.mean() / 1e3

def get_all_percentile(df, percentile):
        p = float('.{}'.format(percentile.split('p')[1]))
        return df.LATENCY.quantile(q=p, interpolation='nearest') / 1e3

def get_models_percentile(df, percentile):
    p = float('.{}'.format(percentile.split('p')[1]))
    return df.groupby('MODEL').LATENCY.quantile(q=p, interpolation='nearest') / 1e3

# input_path is a directory containing 1 .csv file per load point
def parse_triton(input_path, x_feature, percentiles):
    pctl_offset = {pctl:i for i, pctl in enumerate(percentiles)}
    model_indexes = {
        'mobilenetv2-7': 1,
        'densenet-9': 2,
        'googlenet-9': 3,
        'inception_v3': 4,
        'resnet18-v2-7': 5,
        'resnet34-v2-7': 6,
        'resnet50-v2-7': 7,
        'squeezenet1.1-7': 8,
    }

##    print('Loading result files from {}'.format(input_path))

    if not os.path.isdir(input_path):
        print('{} is not a valid directory'.format(input_path))
        sys.exit(1)

    result_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]
    # Sort by sending_rate
    sorted_results = sorted(result_files, key=lambda f: int(f.split('/')[-1].split('_')[1]))
##    print('Found {}'.format(sorted_results))

    x = [] # A vector of load points
    y = [ # first level: array of per-model matrices
            [ # columns: percentiles as requested by the caller
                np.zeros(len(sorted_results)) # rows: load datapoints vector
                for j in range(len(pctl_offset))
            ]
            for i in range(len(model_indexes)+1)
        ]
    for i, f in enumerate(sorted_results):
#       print('processing {}'.format(f))

        df = pd.read_csv(f, delimiter='\t')

        sending_rate = f.split('/')[-1].split('_')[1]
        sigma = f.split('/')[-1].split('_')[2].split('.csv')[0]

        duration = ((max(df.SEND) - min(df.SEND)) / 2195) / 1e6
        if x_feature == 'throughput':
            x.append(df.shape[0] / duration)
        elif x_feature == 'rate':
            x.append(sending_rate)
        else:
            print('Unexpected X axis feature {}'.format(x_feature))

        for percentile in percentiles:
            if percentile == 'Mean':
                all_value = get_all_mean(df)
                model_values = get_models_mean(df)
            else:
                all_value = get_all_percentile(df, percentile)
                model_values = get_models_percentile(df, percentile)

            # Overall percentile-requested latency
            y[0][pctl_offset[percentile]][i] = all_value
            # Per model latency
            for model_name, model_index in model_indexes.items():
                y[model_index][pctl_offset[percentile]][i] = model_values[model_values.index==model_name].values[0]

    return x, y

##parse_triton('/home/maxdml/triton-client/sosp32_results/1.5', 'throughput', ['p99'])
