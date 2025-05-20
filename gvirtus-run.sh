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
  -v ./CMakeLists.txt:/gvirtus/CMakeLists.txt \
  --name gvirtus \
  --runtime=nvidia \
  taslanidis/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04
#   --entrypoint /bin/bash \
#   --gpus all \