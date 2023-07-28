import numpy as np

def parse_input_kelvin(path, xaxis_name, yaxis_names, model_ids):
    yaxis_name2offset = {'Mean': 0, 'p50': 1, 'p90': 2, 'p95': 3, 'p99': 4}

    data = np.genfromtxt(path, delimiter=',')

    if xaxis_name == 'throughput':
        x_data = data[:, 1] / data[:, 2] * 1000000
    else: # sending rate
        x_data = 1000000. / data[:, 0]

    y_data = [[data[:, model_id * 7 + yaxis_name2offset[yaxis_name] + 3] / 1000. for yaxis_name in yaxis_names] for model_id in model_ids]

    return x_data, y_data
