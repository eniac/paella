#!/bin/bash

install_path=/bigdisk/opt/llis
models_dir=/bigdisk/models/cuda
res_dir=/bigdisk/results-mps

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

mkdir -p "$res_dir"

script_path="$(dirname "$0")"

PIDS=()

cleanup() {
    echo *********************** Killing ${PIDS[@]}
    kill ${PIDS[@]}
    echo quit | sudo nvidia-cuda-mps-control
    sudo nvidia-smi -c DEFAULT
    exit
}

trap "cleanup" INT

jobs=(
    "${models_dir}/resnet18-v2-7-cuda-pack.so 119"
    "${models_dir}/inception_v3-cuda-pack.so 6"
)

sudo nvidia-smi -c EXCLUSIVE_PROCESS
sudo nvidia-cuda-mps-control -d

for ln_sigma in {1.5,2}; do
    for i in {1818,2000,2222,2500,2857,3333,4000,5000,6667,10000,20000,40000}; do
        $install_path/bin/workload_pregen \
            --iat $i \
            --ln_sigma $ln_sigma \
            --seed 1 \
            --pregen_prefix "${res_dir}/resnet18_inception_v3_prop_mps_pregen" \
            --num_jobs 3000 \
            0.952 \
            0.048

        PIDS=()

        start_time=$(($(date +%s%N)+10000000000))

        for job_id in "${!jobs[@]}"; do
            set -- ${jobs[$job_id]}
            sudo ${install_path}/bin/tvm_direct_multistream_pregen \
                --iat $i \
                --ln_sigma $ln_sigma \
                --start_record_num 0 \
                --seed 1 \
                --prefix "${res_dir}/resnet18_inception_v3_prop_mps" \
                --pregen_prefix "${res_dir}/resnet18_inception_v3_prop_mps_pregen" \
                --pregen_job_id ${job_id} \
                --preset_start_time $start_time \
                --iat_n \
                --iat_g \
                --ln_sigma_n \
                --num_jobs 3000 \
                --concurrency $2 \
                $1 $2 &
            PIDS+=($!)
        done
        
        wait

        python3 "${script_path}/tools/merge_mps_results.py" --prefix "${res_dir}/resnet18_inception_v3_prop_mps_lns${ln_sigma}" --iat $i --num_jobs 2
    done
done

echo quit | sudo nvidia-cuda-mps-control
sudo nvidia-smi -c DEFAULT
