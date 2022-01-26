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

echo "**** Running ultraface_arcface with ln_sigma=$ln_sigma, suffix=$suffix"

for seed in {1,}; do
    #for i in {2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600}; do
    #for i in {17000,20000,25000,33000,50000,100000}; do
    for i in {25000,50000,100000}; do
        taskset -c 4 ./server server 1000000 1 &
        SERVER_PID=$!
        sleep 5

        ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i $ln_sigma 30 3000 0 ultraface_arcface_0.7_0.3${suffix}.txt tmp2.txt ultraface_arcface_0.7_0.3${suffix}_profile_$i ultraface_arcface_0.7_0.3${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_ultraface320/libjob_tvm_ultraface320.so" 0.7 30 "${abs_path}/release/jobs/tvm_arcfaceresnet100/libjob_tvm_arcfaceresnet100.so" 0.3 30
        wait
    done
done
