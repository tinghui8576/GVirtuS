#!/bin/bash

apt update
apt install -y nvidia-driver-580

# mkdir -p /gvirtus/build && cd /gvirtus/build && cmake .. && make -j$(nproc) && make install

apt update && apt install -y python3 python3-pip
# apt install -y python3-pip python-is-python3
pip3 install torch torchvision pillow

# cd /gvirtus/examples/torchvision-pretrained

export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib
export GVIRTUS_LOGLEVEL=5000
LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcuda.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcublas.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcublasLt.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcudnn.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcufft.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcurand.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcusparse.so: \
    ${GVIRTUS_HOME}/lib/frontend/libcusolver.so: \
    ${GVIRTUS_HOME}/lib/frontend/libnvrtc.so" \
# python3 classify.py 
# > frontend.log 2>&1
python3 graph.py
# LD_DEBUG=libs python3 -c "import torch; torch.cuda.is_available()" 2>&1 | grep -m1 'libcuda.so.1'