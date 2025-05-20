#! /bin/bash
export GVIRTUS_HOME=/usr/local/gvirtus
export GVIRTUS_LOGLEVEL=0
mkdir gvirtus/build && cd gvirtus/build && cmake .. && make && make install
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}
sed -i 's/"server_address": "127.0.0.1"/"server_address": "0.0.0.0"/' ${GVIRTUS_HOME}/etc/properties.json
${GVIRTUS_HOME}/bin/gvirtus-backend ${GVIRTUS_HOME}/etc/properties.json