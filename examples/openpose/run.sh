export OPENPOSE_ROOT=/root/openpose
export LD_LIBRARY_PATH=${OPENPOSE_ROOT}/build/src/openpose:${GVIRTUS_HOME}/lib/:${GVIRTUS_HOME}/lib/frontend
nvcc 00_test.cpp -g -o 00_test \
-I${OPENPOSE_ROOT}/include \
-I${OPENPOSE_ROOT}/3rdparty/caffe/include \
-L${OPENPOSE_ROOT}/build/src/openpose \
-L${OPENPOSE_ROOT}/build/caffe/lib \
 -lopenpose -lcaffe -lgflags \
`pkg-config --cflags --libs opencv4`

cd ${OPENPOSE_ROOT} && /gvirtus/examples/openpose/00_test