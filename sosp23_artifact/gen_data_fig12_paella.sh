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

for ln_sigma in {1.5,2}; do
    for sched in {SS,MS-jbj,MS-kbk,Full}; do
        case $sched in
            SS)
                sched=fifo
                num_streams=1
                suffix=_ss
                ;;
            MS-jbj)
                sched=fifo
                num_streams=500
                suffix=_msjbj
                ;;
            MS-kbk)
                sched=fifo2
                num_streams=500
                suffix=_mskbk
                ;;
            Full)
                sched=full3
                num_streams=500
                suffix=_full
                ;;
        esac

        for i in {2222,2500,2857,3333,4000,5000,6667,10000,20000,40000}; do
            taskset -c 4 "${install_path}"/bin/llis_server \
                --name server \
                --sched $sched \
                --num_streams $num_streams &
            SERVER_PID=$!
            sleep 5

            "${install_path}"/bin/llis_app_client \
                --server_name server \
                --iat $i \
                --ln_sigma $ln_sigma \
                --start_record_num 0 \
                --seed 1 \
                --prefix "${res_dir}/resnet18_inception_v3_prop${suffix}" \
                --fairness 1000000 \
                --iat_n \
                --iat_g \
                --ln_sigma_n \
                --num_jobs 3000 \
                --concurrency 125 \
                "${install_path}/lib/llis_jobs/libjob_tvm_resnet18.so" 0.952 119 \
                "${install_path}/lib/llis_jobs/libjob_tvm_inception_v3.so" 0.048 6
            wait
        done
    done
done
