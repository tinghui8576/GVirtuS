#!/bin/bash
apt update && apt install -y python3 python3-pip
# apt install -y nvidia-driver-550
pip3 install torch==2.8.0+cu126 torchvision pillow --index-url https://download.pytorch.org/whl/cu126
