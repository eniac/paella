#!/bin/bash

res_dir=$1
model_path=$2
ln_sigma=$3
suffix=$4

cd "$(dirname "$0")"/..
abs_path="`pwd`"

cd release/tests/simple

echo "**** Running all-direct with ln_sigma=$ln_sigma, suffix=$suffix"

for seed in {1,}; do
    #for i in {3000,6000,8000,10000,12000,14000,16000,18000,20000,22000,50000,100000,200000,500000}; do
    #for i in {3000,}; do
    #for i in {25000,50000,100000,200000,500000}; do
    for i in {0,}; do
        #ncu -f --set full --profile-from-start on -o "${res_dir}/all_equal_direct${suffix}.ncu" \
        nsys profile -o "${res_dir}/all_equal_direct${suffix}.nsys" \
        ./test_tvm_direct_concurrent \
            --iat $i \
            --ln_sigma $ln_sigma \
            --concurrency 15 \
            --num_jobs 500 \
            --seed $seed \
            --output_path "${res_dir}/all_equal_direct${suffix}.txt" \
            "${model_path}/mnist-8-cuda-pack.so" 0.112 \
            "${model_path}/mobilenetv2-7-cuda-pack.so" 0.111 \
            "${model_path}/densenet-9-cuda-pack.so" 0.111 \
            "${model_path}/googlenet-9-cuda-pack.so" 0.111 \
            "${model_path}/inception_v3-cuda-pack.so" 0.111 \
            "${model_path}/resnet18-v2-7-cuda-pack.so" 0.111 \
            "${model_path}/resnet34-v2-7-cuda-pack.so" 0.111 \
            "${model_path}/resnet50-v2-7-cuda-pack.so" 0.111 \
            "${model_path}/squeezenet1.1-7-cuda-pack.so" 0.111
    done
done

