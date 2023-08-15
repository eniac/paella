## Setup

*Note that you should skip this part if you are using the machine we provided, as it already has the environment set up.*

The following scripts install stuff in the `/bigdisk` directory by default. If you want to change the installation prefix, you can use the `-p` argument (e.g., `./install_dependencies.sh -p /my_dir`).

First, please download the onnx models from [here](https://drive.google.com/file/d/1AAI1lwGBT6CnLx_q24z8vhqKh1g8mT3g/view?usp=sharing) and extract it to `<installation_prefix>/models` (`/bigdisk/models` by default). The setup scripts below depend on the onnx models.

Then, `cd` into the `setup/` directory if you have not done so yet.

1. Run `./install_dependencies.sh` to setup the environment.

1. Run `./install_llis_tvm.sh`, which install Paella and the custom TVM modified for Paella. Also, it will compile the models with TVM.

1. Run `./build_triton_docker.sh` to build the docker with Triton server.

1. Run `./install_triton_client.sh` to install the Triton client.

*After running all scripts, either `source ~/.bash_profile` or logout and then log back in to ensure that the environment variables are set.*

## Reset

We have already done the setup process on the machine we provided. However, if you want to start from scratch, run `./reset_all.sh`.

If you only want to install Paella and the custom TVM from scratch, while keeping other dependencies untouched, run `./reset_llis_tvm.sh` and then run `./install_llis_tvm.sh` again.

*Either `source ~/.bash_profile` or logout and then log back in to ensure that the environment variables are set.*

