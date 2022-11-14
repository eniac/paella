#!/bin/bash

res_dir=$1
ln_sigma=$2
suffix=$3

cd "$(dirname "$0")"/..
abs_path="`pwd`"

cd release/src/server

SERVER_PID=0

trap "kill $SERVER_PID; exit" INT

echo "**** Running all with ln_sigma=$ln_sigma, suffix=$suffix"

for seed in {1,}; do
    #for i in {3000,6000,8000,10000,12000,14000,16000,18000,20000,22000,50000,100000,200000,500000}; do
    for i in {40000,50000,100000,250000,500000}; do
        taskset -c 4 ./server server 1000000 1 &
        SERVER_PID=$!
        sleep 5

        #../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i $ln_sigma 15 3000 0 all_equal${suffix}.txt tmp2.txt all_equal${suffix}_profile_$i all_equal${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 0.125 15 "${abs_path}/release/jobs/tvm_ultraface320/libjob_tvm_ultraface320.so" 0.125 15 "${abs_path}/release/jobs/tvm_mobilenet/libjob_tvm_mobilenet.so" 0.125 15 "${abs_path}/release/jobs/tvm_densenet121/libjob_tvm_densenet121.so" 0.125 15 "${abs_path}/release/jobs/tvm_resnet50/libjob_tvm_resnet50.so" 0.125 15 "${abs_path}/release/jobs/tvm_googlenet/libjob_tvm_googlenet.so" 0.125 15 "${abs_path}/release/jobs/tvm_arcfaceresnet100/libjob_tvm_arcfaceresnet100.so" 0.125 15 "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 0.125 15

        ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi \
            --server_name server \
            --iat $i \
            --ln_sigma $ln_sigma \
            --concurrency 1 \
            --num_jobs 1000 \
            --start_record_num 0 \
            --seed $seed \
            --prefix "${res_dir}/mnist_googlenet_0.7_0.3${suffix}" \
            --fairness 1000000 \
            --iat_n \
            --iat_g \
            --ln_sigma_n \
            "${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 0.7 1 \
            "${abs_path}/release/jobs/tvm_googlenet/libjob_tvm_googlenet.so" 0.3 1
        wait
    done
done
