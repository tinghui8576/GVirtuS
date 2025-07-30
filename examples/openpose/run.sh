nvcc 00_test.cpp -o 00_test -I /root/openpose/include -L /root/openpose/build/src/openpose `pkg-config --cflags --libs opencv4` -lopenpose -lgflags
export LD_LIBRARY_PATH=/root/openpose/build/src/openpose:/usr/local/gvirtus/lib/frontend:$LD_LIBRARY_PATH
./00_test
