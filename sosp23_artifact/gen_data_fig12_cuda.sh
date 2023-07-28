#!/bin/bash

install_path=/bigdisk/opt/llis
models_dir=/bigdisk/models/cuda
res_dir=/bigdisk/results-cuda

while getopts 'p:m:o:' opt; do
  case "$opt" in
    p)
      install_path="$OPTARG"
      ;;

    m)
      models_dir="$OPTARG"
      ;;

    o)
      res_dir="$OPTARG"
      ;;
   
    ?|h)
      echo "Usage: $(basename $0) [-p install_path] [-m model_dir] [-o output_dir]"
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"

export CUDA_DEVICE_MAX_CONNECTIONS=32

mkdir -p "${res_dir}"

for ln_sigma in {1.5,2}; do
    for sched in {CUDA-SS,CUDA-MS}; do
        case $sched in
            CUDA-SS)
                num_streams=1
                suffix=_cudass
                ;;
            CUDA-MS)
                num_streams=141
                suffix=_cudams
                ;;
        esac

        for i in {2000,2222,2500,2857,3333,4000,5000,6667,10000,20000,40000}; do
            "${install_path}"/bin/tvm_direct_multistream \
                --iat $i \
                --ln_sigma $ln_sigma \
                --start_record_num 0 \
                --seed 1 \
                --prefix "${res_dir}/resnet18_inception_v3_prop${suffix}" \
                --iat_n \
                --iat_g \
                --ln_sigma_n \
                --num_jobs 3000 \
                --concurrency $num_streams \
                "${models_dir}/resnet18-v2-7-cuda-pack.so" 0.952 119 \
                "${models_dir}/inception_v3-cuda-pack.so" 0.048 6
        done
    done
done
