#!/bin/bash

res_dir=$1
ln_sigma=$2
suffix=$3

cd "$(dirname "$0")"/..
abs_path="`pwd`"

cd release/src/server

SERVER_PID=0

trap "kill $SERVER_PID; exit" INT

echo "**** Running mnist with ln_sigma=$ln_sigma, suffix=$suffix"

for seed in {1,}; do
    for i in {0,}; do
        #for s in {0,10,100,1000,10000,}; do # 0,0.05,0.4,3,30 us
        #for s in {100000,1000000}; do # ? us
        for s in {0,}; do
            taskset -c 4 ./server server 1000000 1 $s &
            SERVER_PID=$!
            sleep 5

            #../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i $ln_sigma 1 3000 0 mnist_sched_sleep${suffix}.txt tmp2.txt mnist_sched_sleep${s}${suffix}_profile_$i mnist_sched_sleep${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 1 1

            ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi \
                --server_name server \
                --iat $i \
                --ln_sigma $ln_sigma \
                --concurrency 1 \
                --num_jobs 3000 \
                --start_record_num 0 \
                --seed $seed \
                --prefix "${res_dir}/mnist_sched_sleep${suffix}" \
                --fairness 1000000 \
                --sched_sleep $s \
                --sched_sleep_n \
                --sched_sleep_g \
                "${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 1 1
            wait
        done
    done
done
