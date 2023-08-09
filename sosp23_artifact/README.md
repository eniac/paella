# Paella: Low-latency Model Serving with Virtualized GPU Scheduling

This is the artifact for the SOSP 2023 Paper #224 Paella: Low-latency Model Serving with Virtualized GPU Scheduling.

If you are using the machine we provided, there is a copy of this directory in the home directory.

Note that you should verify, e.g., using the `who` or `ps -ef` commands, that other evaluators are not using the machine at the same time or the performance results will include outside contention.

Also note that the machine we provide---while free and publically accessible thanks to the NSF---uses an older GPU model than the one used in our evaluation, which limits features and affects the absolute values of the evaluation but not the trends.

## Setup

Please refer to the [instructions](setup/README.md) in the `setup/` directory to setup the environment.

*Skip this part if you are using the machine we provided, as it already has the environment set up.*

## Serving a mix of 8 models (fig. 11)

### Paella and its ablations

Generate data for Paella and its ablations (3 hours 15 mins). Results are stored at `/bigdisk/results`.

```
./gen_data_fig11_paella.sh
```

### Direct CUDA

Generate data for direct CUDA (CUDA-SS, CUDA-MS) (1 hour 20 mins). Results are stored at `/bigdisk/results-cuda`.

```
./gen_data_fig11_cuda.sh
```

### Triton

Open two terminals. First, on one terminal, run the following to start the triton server:

```
./triton_server_launch.sh
```

Then, on another terminal, run the following to generate data for Triton (10 mins). Results are stored at `/bigdisk/results-triton`.

```
./gen_data_fig11_triton.sh
```

*Stop the Triton server after it finishes* by Ctrl-C on the terminal running it. Otherwise, it will affect the results of other experiments.

### Plots

To plot the graphs, run the following:

```
./plot_fig11.sh
```

It generates two graphs at `/bigdisk/graphs`. `fig11_lns1.5.pdf` is for $\sigma=1.5$ and `fig11_lns2.pdf` is for $\sigma=2$.

## Serving 2 extreme models (fig. 12)

### Paella and its ablations

Generate data for Paella and its ablations (1 hour 5 mins). Results are stored at `/bigdisk/results`.

```
./gen_data_fig12_paella.sh
```

### Direct CUDA

Generate data for direct CUDA (CUDA-SS, CUDA-MS) (25 mins). Results are stored at `/bigdisk/results-cuda`.

```
./gen_data_fig12_cuda.sh
```

### MPS

NOTE(MPS): The CloudLab machine's Tesla P100 GPU is pre-Volta and, therefore, does not support full MPS functionality.  We are not including support for this baseline in the artifact evaluation, but theoretically, on a newer GPU, one can generate data for MPS with the following command. Results are stored at `/bigdisk/results-mps`.

```
./gen_data_fig12_mps.sh
```

### Plots

To plot the graphs, run the following:

```
./plot_fig12.sh
```

It generates two graphs at `/bigdisk/graphs`. `fig12_lns1.5.pdf` is for $\sigma=1.5$ and `fig12_lns2.pdf` is for $\sigma=2$.

NOTE(MPS): this script has MPS commented out because of the above issue.  On a GPU with MPS support, you can uncomment the MPS code in the plot script.

## Fairness between 2 extreme models (fig. 13)

### Run

Generate data for Paella when fairness threshold is adjusted. Results are stored at `/bigdisk/results`.

```
./gen_data_fig13.sh
```

### Plot

To plot the graph, run the following:

```
./plot_fig13.sh
```

It generates `/bigdisk/graphs/fig13.pdf`.

## Comparing results with the graphs on the manuscript

We used NVIDIA Tesla T4 for the manuscript. However, the CloudLab machine we use for artifact evaluation uses NVIDIA Tesla P100 instead. So, the values are not directly comparable, but the trends hold.
