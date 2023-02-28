#!/bin/bash

res_dir=$1
model_path=$2
ln_sigma=$3
suffix=$4

cd "$(dirname "$0")"/..
abs_path="`pwd`"

cd release/tests/simple

echo "**** Running all-direct-multistream with ln_sigma=$ln_sigma, suffix=$suffix"

for seed in {1,}; do
    #for i in {3000,6000,8000,10000,12000,14000,16000,18000,20000,22000,50000,100000,200000,500000}; do
    #for i in {3000,}; do
    #for i in {25000,50000,100000,200000,500000}; do
    #for i in {0,}; do
    #for i in {143,154,167,182,200,222,250,286,333,400,500,667,1000,1053,1111,1176,1250,1333,1429,1538,1667,1818,2000}; do
    for i in {2222,2500,2857,3333,4000,5000,6667,10000,20000}; do
        #ncu -f --set full --profile-from-start on -o "${res_dir}/all_equal_direct${suffix}.ncu" \
        #nsys profile -o "${res_dir}/all_equal_direct${suffix}.nsys" \
        ./test_tvm_direct_multistream \
            --iat $i \
            --ln_sigma $ln_sigma \
            --start_record_num 0 \
            --seed $seed \
            --prefix "${res_dir}/all_equal_direct_multistream${suffix}" \
            --iat_n \
            --iat_g \
            --ln_sigma_n \
            --num_jobs 15000 \
            --concurrency 641 \
            "${model_path}/mnist-8-cuda-pack.so" 0.759 487 \
            "${model_path}/mobilenetv2-7-cuda-pack.so" 0.0636 41 \
            "${model_path}/densenet-9-cuda-pack.so" 0.024 15 \
            "${model_path}/googlenet-9-cuda-pack.so" 0.00289 2 \
            "${model_path}/inception_v3-cuda-pack.so" 0.00383 2 \
            "${model_path}/resnet18-v2-7-cuda-pack.so" 0.0657 42 \
            "${model_path}/resnet34-v2-7-cuda-pack.so" 0.0382 25 \
            "${model_path}/resnet50-v2-7-cuda-pack.so" 0.0187 12 \
            "${model_path}/squeezenet1.1-7-cuda-pack.so" 0.02408 15
    done
done


