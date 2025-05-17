#! /bin/bash

docker buildx build \
  --platform linux/amd64 \
  --push \
  -t taslanidis/gvirtus:cuda11.8.0-cudnn8-ubuntu22.04 \
  -f docker/Dockerfile .