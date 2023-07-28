## Setup

*Note that you should skip this part if you are using the machine we provided, as it already has the environment set up.*

First, `cd` into the `setup/` directory if you have not done so yet.

1. Run `./install_dependencies.sh` to setup the environment.

2. Run `./install_llis_tvm.sh`, which install Paella and the custom TVM modified for Paella. Also, it will compile the models with TVM.

3. Run `./build_triton_docker.sh` to build the docker with Triton server.

4. Run `./install_triton_client.sh` to install the Triton client.

## Reset

We have already done the setup process on the machine we provided. However, if you want to start from scratch, run `./reset_all.sh`.

If you only want to install Paella and the custom TVM from scratch, while keeping other dependencies untouched, run `./reset_llis_tvm.sh` and then run `./install_llis_tvm.sh` again.

