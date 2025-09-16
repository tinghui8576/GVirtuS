#! /bin/bash
mkdir -p /gvirtus/build && cd /gvirtus/build && cmake .. && make -j$(nproc) && make install
${GVIRTUS_HOME}/bin/gvirtus-backend ${GVIRTUS_HOME}/etc/properties.json
