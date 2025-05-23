#! /bin/bash

docker run \
  --rm \
  -it \
  -v ./cmake:/gvirtus/cmake/ \
  -v ./etc:/gvirtus/etc/ \
  -v ./include:/gvirtus/include/ \
  -v ./plugins:/gvirtus/plugins/ \
  -v ./src:/gvirtus/src/ \
  -v ./tools:/gvirtus/tools/ \
  -v ./tests:/gvirtus/tests/ \
  -v ./CMakeLists.txt:/gvirtus/CMakeLists.txt \
  -v ./entrypoint.sh:/entrypoint.sh \
  --entrypoint /entrypoint.sh \
  --name gvirtus \
  --runtime=nvidia \
  taslanidis/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04