#!/bin/bash

cd "$(dirname "$0")"
script_path=`pwd`
cd -

paella_res_dir=/bigdisk/results
output_dir=/bigdisk/graphs

while getopts 'p:c:t:o:' opt; do
  case "$opt" in
    p)
      paella_res_dir="$OPTARG"
      ;;

    o)
      output_dir="$OPTARG"
      ;;
   
    ?|h)
      echo "Usage: $(basename $0) [-p paella_res_dir] [-o output_dir]"
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"

mkdir -p /bigdisk/graphs

python3 $script_path/tools/plot_latency_fairness_threshold.py \
    -o $output_dir/fig13.pdf \
    -i $paella_res_dir/resnet18_inception_v3_prop_full_fairness_lns2.txt \
    -a 'Paella' \
    -l 1 -n ResNet-18 \
    -l 2 -n InceptionV3 \
    --yaxis Mean

