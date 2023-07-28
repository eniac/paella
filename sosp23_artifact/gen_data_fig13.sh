#!/bin/bash

install_path=/bigdisk/opt/llis
export LLIS_MODELS_DIR=/bigdisk/models/cuda_llis
res_dir=/bigdisk/results

while getopts 'p:m:o:' opt; do
  case "$opt" in
    p)
      install_path="$OPTARG"
      ;;

    m)
      export LLIS_MODELS_DIR="$OPTARG"
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

SERVER_PID=0

trap "kill $SERVER_PID; exit" INT

for f in {0.01,0.1,1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,10000,100000}; do
    taskset -c 4 "${install_path}"/bin/llis_server \
        --name server \
        --sched full3 \
        --unfair $f \
        --num_streams 500 &
    SERVER_PID=$!
    sleep 5

    "${install_path}"/bin/llis_app_client \
        --server_name server \
        --iat 0 \
        --ln_sigma 2 \
        --start_record_num 0 \
        --seed 1 \
        --prefix "${res_dir}/resnet18_inception_v3_prop_full_fairness" \
        --fairness $f \
        --fairness_n \
        --fairness_g \
        --ln_sigma_n \
        --num_jobs 3000 \
        --concurrency 125 \
        "${install_path}/lib/llis_jobs/libjob_tvm_resnet18.so" 0.952 119 \
        "${install_path}/lib/llis_jobs/libjob_tvm_inception_v3.so" 0.048 6
    wait
done

