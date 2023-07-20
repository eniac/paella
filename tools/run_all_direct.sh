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
    #for i in {0,}; do
    for i in {143,154,167,182,200,222,250,286,333,400,500,667,1000,1053,1111,1176,1250,1333,1429,1538,1667,1818,2000}; do
        #ncu -f --set full --profile-from-start on -o "${res_dir}/all_equal_direct${suffix}.ncu" \
        #nsys profile -o "${res_dir}/all_equal_direct${suffix}.nsys" \
        ./test_tvm_direct_concurrent \
            --iat $i \
            --ln_sigma $ln_sigma \
            --seed $seed \
            --output_path "${res_dir}/all_equal_direct${suffix}.txt" \
            --num_jobs 15000 \
            --concurrency 60 \
            "${model_path}/mnist-8-cuda-pack.so" 0.759 \
            "${model_path}/mobilenetv2-7-cuda-pack.so" 0.0636 \
            "${model_path}/densenet-9-cuda-pack.so" 0.024 \
            "${model_path}/googlenet-9-cuda-pack.so" 0.00289 \
            "${model_path}/inception_v3-cuda-pack.so" 0.00383 \
            "${model_path}/resnet18-v2-7-cuda-pack.so" 0.0657 \
            "${model_path}/resnet34-v2-7-cuda-pack.so" 0.0382 \
            "${model_path}/resnet50-v2-7-cuda-pack.so" 0.0187 \
            "${model_path}/squeezenet1.1-7-cuda-pack.so" 0.02408
            #--num_jobs 500 \
            #--concurrency 15 \
            #"${model_path}/mnist-8-cuda-pack.so" 0.112 \
            #"${model_path}/mobilenetv2-7-cuda-pack.so" 0.111 \
            #"${model_path}/densenet-9-cuda-pack.so" 0.111 \
            #"${model_path}/googlenet-9-cuda-pack.so" 0.111 \
            #"${model_path}/inception_v3-cuda-pack.so" 0.111 \
            #"${model_path}/resnet18-v2-7-cuda-pack.so" 0.111 \
            #"${model_path}/resnet34-v2-7-cuda-pack.so" 0.111 \
            #"${model_path}/resnet50-v2-7-cuda-pack.so" 0.111 \
            #"${model_path}/squeezenet1.1-7-cuda-pack.so" 0.111
    done
done

