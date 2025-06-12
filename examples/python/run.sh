#LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so:${GVIRTUS_HOME}/lib/frontend/libcuda.so:${GVIRTUS_HOME}/lib/frontend/libcublas.so:${GVIRTUS_HOME}/lib/frontend/libcudnn.so:${GVIRTUS_HOME}/lib/frontend/libcufft.so:${GVIRTUS_HOME}/lib/frontend/libcurand.so:${GVIRTUS_HOME}/lib/frontend/libcusparse.so:${GVIRTUS_HOME}/lib/frontend/libcusolver.so:${GVIRTUS_HOME}/lib/frontend/libnvrtc.so" \
#LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib \
#PYTORCH_NVML_BASED_CUDA_CHECK=1 \
#TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
#python3 distilbert.py

LD_PRELOAD="${GVIRTUS_HOME}/lib/frontend/libcudart.so:${GVIRTUS_HOME}/lib/frontend/libcuda.so:${GVIRTUS_HOME}/lib/frontend/libcublas.so:${GVIRTUS_HOME}/lib/frontend/libcudnn.so:${GVIRTUS_HOME}/lib/frontend/libcufft.so:${GVIRTUS_HOME}/lib/frontend/libcurand.so:${GVIRTUS_HOME}/lib/frontend/libcusparse.so:${GVIRTUS_HOME}/lib/frontend/libcusolver.so:${GVIRTUS_HOME}/lib/frontend/libnvrtc.so" \
LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${GVIRTUS_HOME}/lib \
PYTORCH_NVML_BASED_CUDA_CHECK=1 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python3 distilbert.py
#strace -f -e trace=openat,mmap,mprotect,read python3 distilbert.py -o strace.log
#ltrace -f -e '__cudaRegisterFatBinary+__cudaRegisterVar+__cudaRegisterFunction+cudaMalloc+cudaMemcpy+cudaLaunchKernel' python3 distilbert.py
#ltrace -f -o ltrace.log python3 distilbert.py
#LD_AUDIT=./audit.so python3 distilbert.py
#LD_DEBUG=libs python3 distilbert.py

#objdump -T /usr/local/gvirtus/lib/frontend/libcudart.so | grep __cudaRegisterFatBinary
#nm -D /usr/local/gvirtus/lib/frontend/libcudart.so | grep __cudaRegisterFatBinary