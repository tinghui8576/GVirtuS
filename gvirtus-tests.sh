#! /bin/bash

docker exec -it gvirtus bash -c 'export LD_LIBRARY_PATH=$GVIRTUS_HOME/lib/frontend:$LD_LIBRARY_PATH && cd /gvirtus/build && ctest --output-on-failure'