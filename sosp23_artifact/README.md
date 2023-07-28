# Paella: Low-latency Model Serving with Virtualized GPU Scheduling

This is the artifact for the SOSP 2023 Paper #224 Paella: Low-latency Model Serving with Virtualized GPU Scheduling.

## Setup

Please refer to the [instructions](setup/README.md) in the `setup/` directory to setup the environment.

*Note that you should skip this part if you are using the machine we provided, as it already has the environment set up.*

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

### Plots

To plot the graphs, run the following:

```
./plot_fig12.sh
```

It generates two graphs at `/bigdisk/graphs`. `fig12_lns1.5.pdf` is for $\sigma=1.5$ and `fig12_lns2.pdf` is for $\sigma=2$.

