#!/bin/bash

cd "$(dirname "$0")"
script_path=`pwd`
cd -

paella_res_dir=/bigdisk/results
cuda_res_dir=/bigdisk/results-cuda
triton_res_dir=/bigdisk/results-triton/
output_dir=/bigdisk/graphs

while getopts 'p:c:t:o:' opt; do
  case "$opt" in
    p)
      paella_res_dir="$OPTARG"
      ;;

    c)
      cuda_res_dir="$OPTARG"
      ;;

    t)
      triton_res_dir="$OPTARG"
      ;;

    o)
      output_dir="$OPTARG"
      ;;
   
    ?|h)
      echo "Usage: $(basename $0) [-p paella_res_dir] [-c cuda_res_dir] [-t triton_res_dir] [-o output_dir]"
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"

mkdir -p ${output_dir}

python3 $script_path/tools/plot_latency_throughput_subplots.py \
    -o $output_dir/fig11_lns2.pdf \
    -i $cuda_res_dir/all_prop_cudass_lns2.txt \
    -a 'CUDA-SS' \
    -i $cuda_res_dir/all_prop_cudams_lns2.txt \
    -a 'CUDA-MS' \
    -i $triton_res_dir/2 \
    -a 'Triton' \
    -i $paella_res_dir/all_prop_ss_lns2.txt \
    -a 'Paella-SS' \
    -i $paella_res_dir/all_prop_msjbj_lns2.txt \
    -a 'Paella-MS-jbj' \
    -i $paella_res_dir/all_prop_mskbk_lns2.txt \
    -a 'Paella-MS-kbk' \
    -i $paella_res_dir/all_prop_full_lns2.txt \
    -a 'Paella' \
    -m 5 -n ResNet-18 \
    -m 1 -n MobileNetV2 \
    -m 6 -n ResNet-34 \
    -m 8 -n SqueezeNet1.1 \
    -m 0 -n All \
    -m 7 -n ResNet-50 \
    -m 2 -n Densenet \
    -m 3 -n GoogleNet \
    -m 4 -n InceptionV3 \
    --xaxis throughput \
    --yaxis p99 \
    --subplotx 2 \
    --subploty 5 \
    --legend_subplot 9 \
    --skip_subplot 9 \
    --no-xlabel \
    --width 10.7 \
    --height 3 \
    --ylim 75

python3 $script_path/tools/plot_latency_throughput_subplots.py \
    -o $output_dir/fig11_lns1.5.pdf \
    -i $cuda_res_dir/all_prop_cudass_lns1.5.txt \
    -a 'CUDA-SS' \
    -i $cuda_res_dir/all_prop_cudams_lns1.5.txt \
    -a 'CUDA-MS' \
    -i $triton_res_dir/1.5 \
    -a 'Triton' \
    -i $paella_res_dir/all_prop_ss_lns1.5.txt \
    -a 'Paella-SS' \
    -i $paella_res_dir/all_prop_msjbj_lns1.5.txt \
    -a 'Paella-MS-jbj' \
    -i $paella_res_dir/all_prop_mskbk_lns1.5.txt \
    -a 'Paella-MS-kbk' \
    -i $paella_res_dir/all_prop_full_lns1.5.txt \
    -a 'Paella' \
    -m 5 -n ResNet-18 \
    -m 1 -n MobileNetV2 \
    -m 6 -n ResNet-34 \
    -m 8 -n SqueezeNet1.1 \
    -m 0 -n All \
    -m 7 -n ResNet-50 \
    -m 2 -n Densenet \
    -m 3 -n GoogleNet \
    -m 4 -n InceptionV3 \
    --xaxis throughput \
    --yaxis p99 \
    --subplotx 2 \
    --subploty 5 \
    --legend_subplot 9 \
    --skip_subplot 9 \
    --width 10.7 \
    --height 3 \
    --ylim 75
