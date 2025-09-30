#!/bin/bash
apt update && apt install -y python3 python3-pip
# apt install -y nvidia-driver-550
pip3 install torch torchvision pillow
