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
    #for i in {133,143,154,167,182,200,222,250,286,333,400,500,667,1000,1053,1111,1176,1250,1333,1429,1538,1667,1818,2000}; do
    #for i in {500,667,1000,1053,1111,1176,1250,1333,1429,1538,1667,1818,2000}; do
    #for i in {1176,1250,1333,1429,1538,1667,1818,2000}; do
    for i in {1000,1053,1111,1176,1250,1333,1429,1538,1667,1818,2000,2222,2500,2857,3333,4000,5000,6667,10000,20000,40000,80000,160000}; do
        #nsys profile -o "${res_dir}/all_equal${suffix}.nsys" \
        taskset -c 4 ./server server 1000000 1 &
        SERVER_PID=$!
        sleep 5

        #../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi server $i $ln_sigma 15 3000 0 all_equal${suffix}.txt tmp2.txt all_equal${suffix}_profile_$i all_equal${suffix}_timeline${i}.txt $seed "${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 0.125 15 "${abs_path}/release/jobs/tvm_ultraface320/libjob_tvm_ultraface320.so" 0.125 15 "${abs_path}/release/jobs/tvm_mobilenet/libjob_tvm_mobilenet.so" 0.125 15 "${abs_path}/release/jobs/tvm_densenet121/libjob_tvm_densenet121.so" 0.125 15 "${abs_path}/release/jobs/tvm_resnet50/libjob_tvm_resnet50.so" 0.125 15 "${abs_path}/release/jobs/tvm_googlenet/libjob_tvm_googlenet.so" 0.125 15 "${abs_path}/release/jobs/tvm_arcfaceresnet100/libjob_tvm_arcfaceresnet100.so" 0.125 15 "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 0.125 15

        ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi \
            --server_name server \
            --iat $i \
            --ln_sigma $ln_sigma \
            --start_record_num 0 \
            --seed $seed \
            --prefix "${res_dir}/all_prop${suffix}" \
            --fairness 1000000 \
            --iat_n \
            --iat_g \
            --ln_sigma_n \
            --num_jobs 3000 \
            --concurrency 187 \
            "${abs_path}/release/jobs/tvm_mobilenet/libjob_tvm_mobilenet.so" 0.257 48 \
            "${abs_path}/release/jobs/tvm_densenet121/libjob_tvm_densenet121.so" 0.0706 13 \
            "${abs_path}/release/jobs/tvm_googlenet/libjob_tvm_googlenet.so" 0.0546 10 \
            "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 0.0138 3 \
            "${abs_path}/release/jobs/tvm_resnet18/libjob_tvm_resnet18.so" 0.272 51 \
            "${abs_path}/release/jobs/tvm_resnet34/libjob_tvm_resnet34.so" 0.168 31 \
            "${abs_path}/release/jobs/tvm_resnet50/libjob_tvm_resnet50.so" 0.0745 14 \
            "${abs_path}/release/jobs/tvm_squeezenet1_1/libjob_tvm_squeezenet1_1.so" 0.0894999999999999 17
            #--num_jobs 15000 \
            #--concurrency 1886 \
            #"${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 0.901 1699 \
            #"${abs_path}/release/jobs/tvm_mobilenet/libjob_tvm_mobilenet.so" 0.0254 48 \
            #"${abs_path}/release/jobs/tvm_densenet121/libjob_tvm_densenet121.so" 0.00698 13 \
            #"${abs_path}/release/jobs/tvm_googlenet/libjob_tvm_googlenet.so" 0.0054 10 \
            #"${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 0.00136 3 \
            #"${abs_path}/release/jobs/tvm_resnet18/libjob_tvm_resnet18.so" 0.0269 51 \
            #"${abs_path}/release/jobs/tvm_resnet34/libjob_tvm_resnet34.so" 0.0166 31 \
            #"${abs_path}/release/jobs/tvm_resnet50/libjob_tvm_resnet50.so" 0.00737 14 \
            #"${abs_path}/release/jobs/tvm_squeezenet1_1/libjob_tvm_squeezenet1_1.so" 0.00899000000000007 17
            #--num_jobs 3000 \
            #--concurrency 15 \
            #"${abs_path}/release/jobs/tvm_mnist/libjob_tvm_mnist.so" 0.112 15 \
            #"${abs_path}/release/jobs/tvm_mobilenet/libjob_tvm_mobilenet.so" 0.111 15 \
            #"${abs_path}/release/jobs/tvm_densenet121/libjob_tvm_densenet121.so" 0.111 15 \
            #"${abs_path}/release/jobs/tvm_googlenet/libjob_tvm_googlenet.so" 0.111 15 \
            #"${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 0.111 15 \
            #"${abs_path}/release/jobs/tvm_resnet18/libjob_tvm_resnet18.so" 0.111 15 \
            #"${abs_path}/release/jobs/tvm_resnet34/libjob_tvm_resnet34.so" 0.111 15 \
            #"${abs_path}/release/jobs/tvm_resnet50/libjob_tvm_resnet50.so" 0.111 15 \
            #"${abs_path}/release/jobs/tvm_squeezenet1_1/libjob_tvm_squeezenet1_1.so" 0.111 15
        wait
    done
done
