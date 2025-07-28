
nvcc main.cu -o main -g `pkg-config --cflags --libs opencv4` -lcublas -lcublasLt -lcudnn
LD_LIBRARY_PATH=/usr/local/gvirtus/lib/frontend:$LD_LIBRARY_PATH gdb ./main
