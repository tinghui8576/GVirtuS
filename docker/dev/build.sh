#! /bin/bash
mkdir -p /gvirtus/build && cd /gvirtus/build && cmake .. && make -j$(nproc) && make install
bash