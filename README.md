# A GPGPU Transparent Virtualization Component for High Performance Computing Clouds

The GPU Virtualization Service (GVirtuS) presented in this work tries to fill the gap between in-house hosted computing clusters, equipped with GPGPUs devices, and pay-for-use high performance virtual clusters deployed via public or private computing clouds. gVirtuS allows an instanced virtual machine to access GPGPUs in a transparent and hypervisor independent way, with an overhead slightly greater than a real machine/GPGPU setup. The performance of the components of gVirtuS is assessed through a suite of tests in different deployment scenarios, such as providing GPGPU power to cloud computing based HPC clusters and sharing remotely hosted GPGPUs among HPC nodes.

**[Click here to read the official GVirtuS paper.](https://link.springer.com/chapter/10.1007/978-3-642-15277-1_37)**

## [How to cite GVirtuS in your scientific papers](CITATIONS.md)

# How To install GVirtuS framework and plugins
## Prerequisites:

* **CUDA Toolkit:** version 12

* **[cuDNN](https://developer.nvidia.com/cudnn-downloads)**: version 8

* [Docker](https://docs.docker.com/engine/install/)

## Installation:

1) `git clone` the **GVirtuS** main repository: 

```   
git clone https://github.com/tgasla/GVirtuS.git
```

2) CD into the repo directory:

```
cd gvirtus
```

2) Checkout to the CUDA 12 branch

```
git checkout cuda-12.6.3
```

## Testing GVirtuS:

To test GVirtuS, follow the steps below. This setup runs the GVirtuS backend inside a Docker container with all dependencies pre-installed, and mounts your local source and test files into the container for easy development and debugging.

**1. Start the GVirtuS Backend**

Use the script below to start the GVirtuS backend. It builds GVirtuS from source inside a Docker container and launches the backend process:

```
make run-gvirtus-backend-dev
```

**2. Run the Tests**

Once the backend is running, you can run the tests using the following script. This script creates a new process inside the same container that acts as the frontend and runs all test files located in the tests/ directory:

```
make run-gvirtus-tests
```

**3. Adding Tests**

To add new tests, simply place your test code in any existing .cu file inside the tests directory. You can also create new .cu files if you wish; just make sure to include them as source files in [tests/CMakeLists.txt](tests/CMakeLists.txt#L32).

**4. Updating and Restarting**

After making local changes to the source or tests:

Stop the currently running GVirtuS backend:

```
make stop-gvirtus
```

Ensure your changes are saved.

Restart the backend and re-run the tests using the scripts above.

> [!NOTE]
> The GVirtuS backend and frontend communicate over localhost (127.0.0.1), so both processes must run on the same machine.

> [!NOTE]
> The gvirtus-run.sh script starts a Docker container with all necessary GVirtuS dependencies and mounts your local source and test directories. This means your local changes are automatically used inside the container, making development and testing fast and efficient—no need to git push or docker push.

> [!IMPORTANT]
> If you make **any changes** to the test files, you must restart the GVirtuS backend using `make stop-gvirtus` followed by `make run-gvirtus-backend-dev`. Otherwise, your test changes will not be picked up.