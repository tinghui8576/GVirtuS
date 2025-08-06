#! /bin/bash
mkdir gvirtus/build && cd gvirtus/build && cmake .. && make -j$(nproc) && make install
${GVIRTUS_HOME}/bin/gvirtus-backend ${GVIRTUS_HOME}/etc/properties.json &
sleep 1
export LD_LIBRARY_PATH=$GVIRTUS_HOME/lib/frontend:$LD_LIBRARY_PATH
ctest --output-on-failure
