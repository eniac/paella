#!/bin/bash

trap "exit" INT

res_dir=$1
suffix=$2

cd ..
abs_path="`pwd`"
cd -

cd ../release/src/server

for seed in {1,}; do
    #for i in {2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000}; do
    for i in {3200,}; do
        taskset -c 4 ./server server 3 1 &
        sleep 5

        #../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i 50 3000 1000 mnist_resnet50_prop${suffix}.txt tmp2.txt mnist_resnet50_prop${suffix}_profile_$i mnist_resnet50_prop${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 0.98 50 "${abs_path}/release/jobs/tvm_resnet50/libjob_tvm_resnet50.so" 0.02 50
        ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i 50 3000 1000 mnist_resnet50_half${suffix}.txt tmp2.txt mnist_resnet50_half${suffix}_profile_$i mnist_resnet50_half${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 0.5 50 "${abs_path}/release/jobs/tvm_resnet50/libjob_tvm_resnet50.so" 0.5 50
        wait
    done
done
