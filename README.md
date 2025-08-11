# GVirtuS

The GPU Virtualization Service (GVirtuS) presented in this work tries to fill the gap between in-house hosted computing clusters, equipped with GPGPUs devices, and pay-for-use high performance virtual clusters deployed via public or private computing clouds. gVirtuS allows an instanced virtual machine to access GPGPUs in a transparent and hypervisor independent way, with an overhead slightly greater than a real machine/GPGPU setup. The performance of the components of gVirtuS is assessed through a suite of tests in different deployment scenarios, such as providing GPGPU power to cloud computing based HPC clusters and sharing remotely hosted GPGPUs among HPC nodes.

**Read the official GVirtuS paper [here](https://link.springer.com/chapter/10.1007/978-3-642-15277-1_37).**

# 📄 Published Papers

You can view the full list of all GVirtuS published papers in [CITATIONS](CITATIONS.md).

# Install from Source Directly on a Host Machine

## Prerequisites

**Tested OS**: Ubuntu 22.04 LTS
    - In principle, every OS that can support CUDA, can also run GVirtuS

Before proceeding, ensure the following dependencies are installed on your system:

```bash
sudo apt update && sudo apt install -y build-essential libxmu-dev libxi-dev libgl-dev libosmesa6-dev liblog4cplus-dev librdmacm-dev libibverbs-dev libgtest-dev liblz4-dev cmake git
```

* [CUDA Toolkit & CUDA Driver](https://developer.nvidia.com/cuda-downloads): _Tested with CUDA Toolkit **v12.6.3** (latest verified working version)_

* [cuDNN](https://developer.nvidia.com/cudnn-downloads): _Tested with cuDNN **v9.5.1** (latest verified working version)_

## Clone the Repository and `cd` into the GVirtuS folder

```bash
git clone https://github.com/tgasla/gvirtus.git && cd gvirtus
```

## Create a build folder and `cd` into it

```bash
mkdir build && cd build
```

## Setup the GVirtuS Environment Variables

### GVIRTUS_HOME

GVirtuS will be installed in the `GVIRTUS_HOME` environment variable path. A good place to install GVirtuS is either your home directory or inside `/usr/local/gvirtus`

```bash
export GVIRTUS_HOME=/usr/local/gvirtus
```

### GVIRTUS_LOGLEVEL

GVirtuS uses log4cplus as its logging system. You can set the loglevel to be any of the following:
- 0: for TRACE
- 10000: for DEBUG
- 20000: for INFO
- 30000: for WARN
- 40000: for ERROR
- 50000: for FATAL

For example:

```bash
export GVIRTUS_LOGLEVEL=20000
```

## Install

```bash
cmake .. && make && make install
```

## Edit the properties.json (if needed)

The GVirtuS backend will listen to the endpoint (address:port) specified in `${GVIRTUS_HOME}/etc/properties.json` configuration file. Similarly, the GVirtuS frontend will try to conenct to the endpoint specified in the same file.

You can directly edit this file with the `server_address` and `port` you want using the following commands:

```bash
sed -i 's/"server_address": "0.0.0.0"/"server_address": "DESIRED_SERVER_ADDRESS"/' ${GVIRTUS_HOME}/etc/properties.json
```

and

```bash
sed -i 's/"port": "9999"/"port": "DESIRED_PORT_NUMBER"/' ${GVIRTUS_HOME}/etc/properties.json
```

## Configure and Run the GVirtuS backend

First, set up the `LD_LIBRARY_PATH` environment variable:

```bash
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}
```

Then, to run the GVirtuS backend on a machine that has access to a physical GPU:

```bash
${GVIRTUS_HOME}/bin/gvirtus_backend ${GVIRTUS_HOME}/etc/properties.json
```

The above command will use `${GVIRTUS_HOME}/etc/properties.json` configuration file.

## Configure the GVirtuS frontend

First, set up the `LD_LIBRARY_PATH` environment variable:

```bash
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
```

Then, you can compile and run any application you want that utilizes cuda and GVirtuS will be used.

If the application is written in python, the above is not enough, you will also need to prefix your python application with the `LD_PRELOAD=$(echo ${GVIRTUS_HOME}/lib/frontend/*.so | tr ' ' ':')`.

For example, if you want to run a python script named `app.py`, instead of doing `python3 app.py`, you should do:

```bash
LD_PRELOAD=$(echo ${GVIRTUS_HOME}/lib/frontend/*.so | tr ' ' ':') python3 app.py
```

> [!WARNING]
> If you use a python manager like pyenv, the above will not work because `python3` is not the python binary. It is ASCII text executable managed by pyenv. Instead, you should either turn off pyenv or manually use the real python binary using:

```bash
LD_PRELOAD=$(echo ${GVIRTUS_HOME}/lib/frontend/*.so | tr ' ' ':') /usr/bin/python3 app.py
```

# 🐳 Use the Official GVirtuS Docker Image

GVirtuS has official Docker images in Docker Hub for `linux/amd64` and `linux/arm64` platforms. You can use them to accelerate your workflow. Follow the instructions below:

## Prerequisites

- Install Docker in both your GVirtuS frontend and backend machines.
- Install NVIDIA Container Toolkit only on your GVirtuS backend machines.

To do so, follow the instructions [here](#prerequisites-2).

## Download GVirtuS Docker image

```bash
docker pull gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04
```

In the Docker image, the environment variable `GVIRTUS_HOME` is set to `/usr/local/gvirtus`, indicating that GVirtuS is installed in that directory.

## Change the GVirtuS Backend Server Address and Port Number (if needed)

By default, the Docker image configures the server address as `0.0.0.0` (listening on all interfaces) and the port as `9999`. If these settings are suitable for your environment, you can start the GVirtuS backend immediately with:

```bash
docker run -it --rm --name gvirtus-backend --network host --runtime nvidia gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04
```

If you want your GVirtuS backend server to listen on a different address or port, follow these steps:

- Create a `properties.json` file anywhere in your host machine similar to the one one in `etc/properties.json`.
- Change the `server_address` and `port` fields as needed and save the file.
- Then, assuming you run the `docker run` command in the same directory as your `properties.json` file, you can overrive the original `properrties.json` configuration file inside the container by doing:

```bash
docker run -it --rm --name gvirtus-backend --network --runtime nvidia host -v ./properties.json:/usr/local/gvirtus/etc/properties.json gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04
```

The original image `GVIRTUS_LOGLEVEL` has been set to 20000 (INFO). If you wish to change that, you can do so by overriding the environment variable. For example, if you want to start the GVirtuS backend server using `DEBUG` loglevel you can do:

```bash
docker run -it --rm --name gvirtus-backend --network host --runtime nvidia -v ./properties.json:/usr/local/gvirtus/etc/properties.json -e GVIRTUS_LOGLEVEL=0 gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04
```

More information on the available log levels can be found [here](#gvirtus_loglevel).

## Run your GVirtuS Frontend Application in a Docker Container

First, create a Dockerfile based on GVirtuS image. Your Dockerfile, should follow the template below.

```Dockerfile
FROM gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04

RUN sed -i 's/"server_address": "0.0.0.0"/"server_address": "BACKEND_SERVER_ADDRESS"/' ${GVIRTUS_HOME}/etc/properties.json
RUN sed -i 's/"port": "9999"/"port": "BACKEND_PORT_NUMBER"/' ${GVIRTUS_HOME}/etc/properties.json

RUN apt update && apt install -y <APPLICATION_DEPENDENCIES_IF_ANY>
...
<DO_ANY_OTHER_SETUP_STEPS_NEEDED>
...
RUN <COMPILE_YOUR_APPLICATION_IF_NEEDED>

ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

ENTRYPOINT ["<YOUR_APPLICATION>"]
CMD ["<default_application_argument_1>, <default_application_argument_2>"]
```

- If your application does not require arguments, you can omit the `CMD` directive from your Dockerfile.
- For Python applications, set the `ENTRYPOINT` to include the `LD_PRELOAD` environment variable before the Python executable to ensure GVirtuS frontend libraries are loaded. Refer to the [Configure the GVirtuS frontend](#configure-the-gvirtus-frontend) section for details.

Then, build your docker image using:

```bash
docker build -t gvirtus-app .
```

Run your application using:

```bash
docker run -it --rm --name gvirtus-app gvirtus-app
```

For example, a Dockerfile for a minimal pytorch application is given below:

```bash
# Dockerfile

FROM gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu126 numpy

COPY ./test_pytorch.py /test_pytorch.py
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

RUN sed -i 's/"server_address": "0.0.0.0"/"server_address": "<YOUR_BACKEND_SERVER_ADDRESS>"/' ${GVIRTUS_HOME}/etc/properties.json
RUN sed -i 's/"port": "9999"/"port": "<YOUR_BACKEND_PORT>"/' ${GVIRTUS_HOME}/etc/properties.json

ENTRYPOINT ["/bin/bash", "-c", "LD_PRELOAD=$(echo ${GVIRTUS_HOME}/lib/frontend/*.so | tr ' ' ':') /usr/bin/python3 /test_pytorch.py"]
```

The `test_pytorch.py` is also given below:

```python
# test_pytorch.py

import torch

device = torch.device("cuda")

model = torch.nn.Linear(10, 1).to(device)

x = torch.randn(5, 10).to(device)

output = model(x)

print(output)
```

You can also write a simple bash script that will act as your application's entrypoint and pass it on the `docker run` command using the `--entrypoint` flag.

# Install from Source using Docker

## Prerequisites
* [Docker](https://docs.docker.com/engine/install/): _Ensure Docker Engine is properly installed and running. Latest verified working version **v26.1.3**_
    - On Ubuntu:
        ```bash
        sudo apt install -y docker.io
        sudo usermod -aG docker $USER
        newgrp docker
        ```

* [Docker Buildx Plugin](https://github.com/docker/buildx#installing): _Used in the Makefile targets that build the GVirtuS docker images_
    - On Ubuntu: `sudo apt install -y docker-buildx-plugin`

* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): _Install only on GPU nodes that will run the GVirtuS backend_
    - On Ubuntu: `sudo apt install -y nvidia-container-toolkit`

## `git clone` the **GVirtuS** main repository: 

```
git clone https://github.com/tgasla/gvirtus.git
```

## `cd` into the repo directory:

```
cd gvirtus
```

## Build the GVirtuS docker image locally

```bash
make docker-build-dev-local
```

## Run the GVirtuS image with GPU access (GVirtuS backend)

```bash
make run-gvirtus-dev
```

## Run the GVirtuS image without GPU access (GVirtuS frontend)

```bash
make run-gvirtus-dev-no-gpu
```

These sections contain information that also apply to the Docker installation:
- [Setup GVirtuS Environment Variables](#setup-the-gvirtus-environment-variables)
- [Edit the properties.json configuration file](#edit-the-propertiesjson-if-needed)
- [Configure and run the GVirtuS backend](#configure-and-run-the-gvirtus-backend)
- [Configure the GVirtuS frontend](#configure-the-gvirtus-frontend)

# 📊 GVirtuS Implementation Status

🗂️ Looking for function coverage? Check out the [**STATUS**](./STATUS.md) file for a detailed breakdown of which CUDA functions are:
- 🛠️ Implemented
- 🧪 Tested
- ⚙️ Working

This file tracks progress across major CUDA libraries (e.g., cuBLAS, cuDNN, cuRAND) and helps monitor GVirtuS coverage.

# 🔬 Testing GVirtuS

To test GVirtuS, follow the steps below. This setup runs the GVirtuS backend inside a Docker container with all dependencies pre-installed, and mounts your local source and test files into the container for easy development and debugging.

## Running GVirtuS unit tests

Use the script below to run the GVirtuS tests. It builds GVirtuS from source inside a Docker container and launches the backend process. This script creates a new process inside the same container that acts as the frontend and runs all test files located in the tests/ directory:

```bash
make run-gvirtus-tests
```

## Adding Tests

To add new tests, simply place your test code in any existing .cu file inside the tests directory. You can also create new .cu files if you wish; just make sure to include them as source files in [tests/CMakeLists.txt](tests/CMakeLists.txt#L24).

## Updating and Restarting

After making local changes to the GVirtuS source or tests, you can re-run the `make run-gvirtus-tests` command.

> [!NOTE]
> The `make run-gvirtus-dev-local` command starts a Docker container with all necessary GVirtuS dependencies and mounts your local repository files required. This means your local changes are automatically used inside the container, making development and testing fast and efficient. 

> [!NOTE]
> The `make run-gvirtus-tests` command does **not** start a new container. Instead, it opens a new shell inside the already running backend container and executes the GVirtuS tests there.

# ⚠️ Disclaimers

GVirtuS is currently not production-ready. It is **not thread-safe** and has known **memory leaks**. Use it with caution in experimental or non-critical environments.