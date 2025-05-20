#! /bin/bash

docker buildx build \
  --platform linux/amd64 \
  --push \
  --no-cache \
  -t taslanidis/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04 \
  .