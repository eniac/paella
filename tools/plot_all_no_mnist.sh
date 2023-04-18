#!/bin/bash

cd "$(dirname "$0")"
script_path=`pwd`
cd -

python3 $script_path/plot_latency_throughput_subplots.py \
    -o all_prop_newmix3_no_mnist_lns2.pdf \
    -i all_prop_direct_singlestream_newmix3_no_mnist_2_lns2.txt \
    -a 'CUDA-SS' \
    -i all_prop_direct_multistream_newmix3_no_mnist_1_lns2.txt \
    -a 'CUDA-MS' \
    -i /home/maxdml/allis/experiments/clockwork/fig10/2 \
    -a 'Clockwork' \
    -i /home/maxdml/triton-client/sosp32_results/2 \
    -a 'Triton' \
    -i all_prop_singlestream_newmix3_no_mnist_fifo_2_lns2.txt \
    -a 'Paella-SS' \
    -i all_prop_newmix3_no_mnist_fifo_2_lns2.txt \
    -a 'Paella-MS-jbj' \
    -i all_prop_newmix3_no_mnist_fifo2_2_lns2.txt \
    -a 'Paella-MS-kbk' \
    -i all_prop_newmix3_no_mnist_full3_2_lns2.txt \
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
    --ylim 1000

python3 $1 $script_path/plot_latency_throughput_subplots.py \
    -o all_prop_newmix3_no_mnist_lns1.5.pdf \
    -i all_prop_direct_singlestream_newmix3_no_mnist_2_lns1.5.txt \
    -a 'CUDA-SS' \
    -i all_prop_direct_multistream_newmix3_no_mnist_1_lns1.5.txt \
    -a 'CUDA-MS' \
    -i /home/maxdml/allis/experiments/clockwork/fig10/1.5 \
    -a 'Clockwork' \
    -i /home/maxdml/triton-client/sosp32_results/1.5 \
    -a 'Triton' \
    -i all_prop_singlestream_newmix3_no_mnist_fifo_2_lns1.5.txt \
    -a 'Paella-SS' \
    -i all_prop_newmix3_no_mnist_fifo_2_lns1.5.txt \
    -a 'Paella-MS-jbj' \
    -i all_prop_newmix3_no_mnist_fifo2_2_lns1.5.txt \
    -a 'Paella-MS-kbk' \
    -i all_prop_newmix3_no_mnist_full3_2_lns1.5.txt \
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
    --ylim 1000
