#!/bin/bash
apt update && apt install -y python3 python3-pip 
python3 -m pip install --no-cache-dir opencv-python-headless ultralytics
