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
    #for i in {2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000}; do
    #for i in {30000,}; do
    #for i in {0,}; do
    #for i in {30000,40000,50000,60000,80000,100000}; do
    for i in {120000,140000}; do
        taskset -c 4 ./server server 1000000 1 &
        SERVER_PID=$!
        sleep 5

        #../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i $ln_sigma 50 3000 0 inception_v3${suffix}.txt tmp2.txt inception_v3${suffix}_profile_$i inception_v3${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 1 50
        #../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi \
        #    --server_name server \
        #    --iat $i \
        #    --ln_sigma $ln_sigma \
        #    --concurrency 50 \
        #    --num_jobs 3000 \
        #    --start_record_num 0 \
        #    --seed $seed \
        #    --prefix inception_v3${suffix} \
        #    --fairness 1000000 \
        #    --iat_n \
        #    --iat_g \
        #    --ln_sigma_n \
        #    "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 1 50

        #../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i $ln_sigma 1 3000 0 inception_v3${suffix}.txt tmp2.txt inception_v3${suffix}_profile_$i inception_v3${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 1 1

        ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi \
            --server_name server \
            --iat $i \
            --ln_sigma $ln_sigma \
            --concurrency 1 \
            --num_jobs 3000 \
            --start_record_num 0 \
            --seed $seed \
            --prefix "${res_dir}/inception_v3${suffix}" \
            --fairness 1000000 \
            --iat_n \
            --iat_g \
            --ln_sigma_n \
            --concurrency_n \
            "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 1 1
        wait
    done
done
