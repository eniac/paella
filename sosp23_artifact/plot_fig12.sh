#!/bin/bash

cd "$(dirname "$0")"
script_path=`pwd`
cd -

paella_res_dir=/bigdisk/results
cuda_res_dir=/bigdisk/results-cuda
mps_res_dir=/bigdisk/results-mps
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

    m)
      mps_res_dir="$OPTARG"
      ;;

    t)
      triton_res_dir="$OPTARG"
      ;;

    o)
      output_dir="$OPTARG"
      ;;
   
    ?|h)
      echo "Usage: $(basename $0) [-p paella_res_dir] [-c cuda_res_dir] [-m mps_res_dir] [-t triton_res_dir] [-o output_dir]"
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"

mkdir -p /bigdisk/graphs

python3 $script_path/tools/plot_latency_throughput_subplots.py \
    -o $output_dir/fig12_lns2.pdf \
    -i $cuda_res_dir/resnet18_inception_v3_prop_cudass_lns2.txt \
    -a 'CUDA-SS' \
    -i $cuda_res_dir/resnet18_inception_v3_prop_cudams_lns2.txt \
    -a 'CUDA-MS' \
    -i $paella_res_dir/resnet18_inception_v3_prop_ss_lns2.txt \
    -a 'Paella-SS' \
    -i $paella_res_dir/resnet18_inception_v3_prop_msjbj_lns2.txt \
    -a 'Paella-MS-jbj' \
    -i $paella_res_dir/resnet18_inception_v3_prop_mskbk_lns2.txt \
    -a 'Paella-MS-kbk' \
    -i $paella_res_dir/resnet18_inception_v3_prop_full_lns2.txt \
    -a 'Paella' \
    -m 0 -n All \
    -m 1 -n ResNet-18 \
    -m 2 -n InceptionV3 \
    --xaxis throughput \
    --yaxis p99 \
    --subplotx 1 \
    --subploty 3 \
    --no-xlabel \
    --legend_subplot 2 \
    --ylim 75 \
    --height 1.4
    #-i $mps_res_dir/resnet18_inception_v3_prop_mps_lns2.txt \
    #-a 'MPS' \

python3 $script_path/tools/plot_latency_throughput_subplots.py \
    -o $output_dir/fig12_lns1.5.pdf \
    -i $cuda_res_dir/resnet18_inception_v3_prop_cudass_lns1.5.txt \
    -a 'CUDA-SS' \
    -i $cuda_res_dir/resnet18_inception_v3_prop_cudams_lns1.5.txt \
    -a 'CUDA-MS' \
    -i $paella_res_dir/resnet18_inception_v3_prop_ss_lns1.5.txt \
    -a 'Paella-SS' \
    -i $paella_res_dir/resnet18_inception_v3_prop_msjbj_lns1.5.txt \
    -a 'Paella-MS-jbj' \
    -i $paella_res_dir/resnet18_inception_v3_prop_mskbk_lns1.5.txt \
    -a 'Paella-MS-kbk' \
    -i $paella_res_dir/resnet18_inception_v3_prop_full_lns1.5.txt \
    -a 'Paella' \
    -m 0 -n All \
    -m 1 -n ResNet-18 \
    -m 2 -n InceptionV3 \
    --xaxis throughput \
    --yaxis p99 \
    --subplotx 1 \
    --subploty 3 \
    --no-title \
    --no-legend \
    --ylim 75 \
    --height 1.4
    #-i $mps_res_dir/resnet18_inception_v3_prop_mps_lns1.5.txt \
    #-a 'MPS' \

