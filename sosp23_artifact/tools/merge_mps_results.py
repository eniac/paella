import argparse
import numpy as np

def print_latency_stats(f, latencies):
    mean = np.mean(latencies);

    sd = np.std(latencies, ddof=1)

    p50 = latencies[int(len(latencies) / 2)];
    p90 = latencies[int(len(latencies) * 0.90)];
    p95 = latencies[int(len(latencies) * 0.95)];
    p99 = latencies[int(len(latencies) * 0.99)];
    max_ele = np.max(latencies)

    f.write(',{},{},{},{},{},{},{}'.format(mean, p50, p90, p95, p99, max_ele, sd));

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-p', '--prefix', dest='prefix');
    parser.add_argument('-i', '--iat', dest='iat', type=int);
    parser.add_argument('-n', '--num_jobs', dest='num_jobs', type=int);

    args = parser.parse_args()

    latencies_per_job = []
    aggs = []
    for job_id in range(args.num_jobs):
        latencies_per_job.append(np.sort(np.loadtxt('{}_job{}_iat{}_raw.txt'.format(args.prefix, job_id, args.iat))))
        aggs.append(np.loadtxt('{}_job{}.txt'.format(args.prefix, job_id), delimiter=',', ndmin=2))
    latencies_all = np.sort(np.concatenate(latencies_per_job))

    time_elasped = 0
    num_job_instances = 0
    for agg in aggs:
        for row in range(agg.shape[0]):
            if (agg[row, 0] == args.iat):
                time_elasped = max(time_elasped, agg[row, 2])
                num_job_instances = int(agg[row, 1])
                break

    with open('{}.txt'.format(args.prefix), 'a') as f:
        f.write('{},{},{}'.format(args.iat, num_job_instances, time_elasped))
        print_latency_stats(f, latencies_all)
        for latencies in latencies_per_job:
            print_latency_stats(f, latencies)
        f.write('\n')

