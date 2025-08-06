#!/bin/bash
apt update && apt install -y python3 python3-pip requests
pip install torch==2.6.0 transformers==4.54.1 accelerate==1.9.0
pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/