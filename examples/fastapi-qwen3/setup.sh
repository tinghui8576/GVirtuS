#!/bin/bash
apt update && apt install -y python3 python3-pip
pip3 install torch==2.6.0 transformers==4.54.1 accelerate==1.9.0 requests==2.32.4 fastapi==0.116.1 uvicorn[standard]==0.35.0
pip3 install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/