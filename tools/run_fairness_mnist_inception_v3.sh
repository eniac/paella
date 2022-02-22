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
    for i in {8000,}; do
        #for f in {0.003,0.03,0.3,3,10,15,20,25,30,60,300}; do
        #for f in {0.00003,0.0003}; do
        #for f in {0.03,0.3,3,30,300,3000,30000}; do
        for f in {3,5,10,15,20,25}; do
            taskset -c 4 ./server server $f 1 &
            SERVER_PID=$!
            sleep 5

            ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i $ln_sigma 50 3000 0 mnist_inception_v3_0.7_0.3_i${i}_fairness${suffix}.txt tmp2.txt mnist_inception_v3_0.7_0.3_fair${f}${suffix}_profile_$i mnist_inception_v3_0.7_0.3_fair${f}${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 0.7 50 "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 0.3 50
            wait
        done
    done
done
