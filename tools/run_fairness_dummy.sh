#!/bin/bash

res_dir=$1
ln_sigma=$2
suffix=$3

cd "$(dirname "$0")"/..
abs_path="`pwd`"

cd release/src/server

SERVER_PID=0

trap "kill $SERVER_PID; exit" INT

for seed in {1,}; do
    for i in {0,}; do
        #for f in {0,1,2,3,4,5,10,15,20,1000000}; do
        #for f in {100,1000,10000}; do
        for f in {2000,4000,6000,8000}; do
            taskset -c 4 ./server server $f 1 &
            SERVER_PID=$!
            sleep 5

            ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi \
                --server_name server \
                --iat $i \
                --ln_sigma $ln_sigma \
                --concurrency 100 \
                --num_jobs 3000 \
                --start_record_num 0 \
                --seed $seed \
                --prefix "${res_dir}/dummy_fairness${suffix}" \
                --fairness $f \
                --iat_n \
                --ln_sigma_n \
                --fairness_n \
                --fairness_g \
                "${abs_path}/release/jobs/dummy_short/libjob_dummy_short.so" 0.7 50 \
                "${abs_path}/release/jobs/dummy_long/libjob_dummy_long.so" 0.3 50
            wait
        done
    done
done
