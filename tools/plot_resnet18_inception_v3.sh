#!/bin/bash

cd "$(dirname "$0")"
script_path=`pwd`
cd -

python3 $script_path/plot_latency_throughput_subplots.py \
    -o resnet18_inception_v3_prop_newmix3_lns2.pdf \
    -i resnet18_inception_v3_prop_direct_singlestream_newmix3_3_lns2.txt \
    -a 'CUDA-SS' \
    -i resnet18_inception_v3_prop_direct_multistream_newmix3_1_lns2.txt \
    -a 'CUDA-MS' \
    -i resnet18_inception_v3_prop_mps_newmix3_4_lns2.txt \
    -a 'MPS' \
    -i resnet18_inception_v3_prop_singlestream_newmix3_no_mnist_fifo_2_lns2.txt \
    -a 'Paella-SS' \
    -i resnet18_inception_v3_prop_newmix3_fifo_1_lns2.txt \
    -a 'Paella-MS-jbj' \
    -i resnet18_inception_v3_prop_newmix3_fifo2_1_lns2.txt \
    -a 'Paella-MS-kbk' \
    -i resnet18_inception_v3_prop_newmix3_full3_1_lns2.txt \
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
    --ylim 1000 \
    --height 1.4
    #--height 1.6

python3 $script_path/plot_latency_throughput_subplots.py \
    -o resnet18_inception_v3_prop_newmix3_lns1.5.pdf \
    -i resnet18_inception_v3_prop_direct_singlestream_newmix3_3_lns1.5.txt \
    -a 'CUDA-SS' \
    -i resnet18_inception_v3_prop_direct_multistream_newmix3_1_lns1.5.txt \
    -a 'CUDA-MS' \
    -i resnet18_inception_v3_prop_mps_newmix3_4_lns1.5.txt \
    -a 'MPS' \
    -i resnet18_inception_v3_prop_singlestream_newmix3_no_mnist_fifo_2_lns1.5.txt \
    -a 'Paella-SS' \
    -i resnet18_inception_v3_prop_newmix3_fifo_1_lns1.5.txt \
    -a 'Paella-MS-jbj' \
    -i resnet18_inception_v3_prop_newmix3_fifo2_1_lns1.5.txt \
    -a 'Paella-MS-kbk' \
    -i resnet18_inception_v3_prop_newmix3_full3_1_lns1.5.txt \
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
    --ylim 1000 \
    --height 1.4
    #--height 1.6

