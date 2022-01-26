#!/bin/bash

res_dir=$1
ln_sigma=$2
suffix=$3

cd ..
abs_path="`pwd`"
cd -

cd ../release/src/server

SERVER_PID=0

trap "kill $SERVER_PID; exit" INT

echo "**** Running mnist_inception_v3 with ln_sigma=$ln_sigma, suffix=$suffix"

for seed in {1,}; do
    #for i in {8000,10000,12000,14000,16000,18000,20000}; do
    for i in {25000,33000,50000,100000}; do
        taskset -c 4 ./server server 1000000 1 &
        SERVER_PID=$!
        sleep 5

        ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i $ln_sigma 50 3000 0 mnist_inception_v3_0.7_0.3${suffix}.txt tmp2.txt mnist_inception_v3_0.7_0.3${suffix}_profile_$i mnist_inception_v3_0.7_0.3${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 0.7 50 "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 0.3 50
        wait
    done
done
