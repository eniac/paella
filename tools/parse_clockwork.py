#!/usr/bin/python3

import pandas as pd
import numpy as np
import argparse
import sys
import os

'''
Triton result files are in the format [sending rate]-[sigma]-[label].csv, e.g., 100-1.5-fig10.csv
Columns are [ MODEL   SEND    RECEIVE LATENCY ]
SEND/RECEIVE/LATENCY are in nanoseconds
We return all times in milliseconds
'''

def get_all_mean(df):
    return df.LATENCY.mean() / 1e6

def get_models_mean(df):
    return df.groupby('MODEL').LATENCY.mean() / 1e6

def get_all_percentile(df, percentile):
        p = float('.{}'.format(percentile.split('p')[1]))
        return df.LATENCY.quantile(q=p, interpolation='nearest') / 1e6

def get_models_percentile(df, percentile):
    p = float('.{}'.format(percentile.split('p')[1]))
    return df.groupby('MODEL').LATENCY.quantile(q=p, interpolation='nearest') / 1e6

# input_path is a directory containing 1 .csv file per load point
def parse_clockwork(input_path, x_feature, percentiles):
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

    model_ids = { 0: "densenet-9",1: "googlenet-9",2: "inceptionv3-kelvin",3:"squeezenet1.1-7",4:"mobilenetv2-7",5:"resnet18-v2-7",7:"resnet50-v2-7",6: "resnet34-v2-7"}
    id_to_name =  {v:k for k,v in model_ids.items()}
    id_to_name['inception_v3'] = id_to_name['inceptionv3-kelvin']

##    print('Loading result files from {}'.format(input_path))

    if not os.path.isdir(input_path):
        print('{} is not a valid directory'.format(input_path))
        sys.exit(1)

    result_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]
    # Sort by sending_rate
    sorted_results = sorted(result_files, key=lambda f: float(f.split('/')[-1].split('-')[0]))
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
        print('processing {}'.format(f))

        df = pd.read_csv(f, delimiter='\t')

        df['model_name'] = df.MODEL.apply(lambda x: model_ids[x])
        df = df.set_index('model_name')

        sending_rate = f.split('/')[-1].split('-')[0]
        sigma = f.split('/')[-1].split('-')[1]

        duration = (max(df.SEND) - min(df.SEND)) / 1e9
        if x_feature == 'throughput':
            x.append(df.shape[0] / duration)
        elif x_feature == 'rate':
            x.append(sending_rate)
        else:
            print('Unexpected X axis feature {}'.format(x_feature))

        print(duration)
        print(df.shape[0] / duration)
        print(sending_rate)
        print("========")

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
                model_id = id_to_name[model_name]
                y[model_index][pctl_offset[percentile]][i] = model_values[model_values.index==model_id].values[0]

    return x, y

parse_clockwork('/home/maxdml/allis/experiments/clockwork/fig10/1.5', 'throughput', ['p99'])
