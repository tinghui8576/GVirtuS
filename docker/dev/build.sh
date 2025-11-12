#! /bin/bash
apt-get update 
mkdir -p /gvirtus/build && cd /gvirtus/build && cmake .. && make -j$(nproc) && make install
bash