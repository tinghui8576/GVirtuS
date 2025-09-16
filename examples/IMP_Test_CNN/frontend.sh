export EXTRA_NVCCFLAGS="--cudart=shared"
export GVIRTUS_LOGLEVEL=20000
export GVIRTUS_HOME=/usr/local/gvirtus
export LD_LIBRARY_PATH=$GVIRTUS_HOME/lib/frontend:$LD_LIBRARY_PATH
# nvcc cnn_graph.cu -o example -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcuda --cudart=shared 
# ./example
# nvcc cnn_nograph.cu -o example -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcuda --cudart=shared 
# ./example
rm -f "Batch.csv"
rm -f "BatchGraph.csv"

# nvcc -O3 -lineinfo cnn_batch_graph.cu -o example \
#   -L "${GVIRTUS_HOME}/lib/frontend" -L "${GVIRTUS_HOME}/lib" \
#   -lcuda --cudart=shared
# nsys profile --stats=true --force-overwrite=true \
#   --trace=cuda,nvtx,osrt \
#   --output=nsys_out \
#   ./example 10000
nvcc cnn_batch.cu -o batch -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcuda --cudart=shared 
./batch 1
./batch 2
./batch 4
./batch 8
./batch 10
./batch 100
./batch 1000
./batch 2000
./batch 3000
./batch 4000
./batch 5000
./batch 6000
./batch 7000
./batch 8000
./batch 9000
./batch 10000

nvcc cnn_batch_graph.cu -o batch_graph -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcuda --cudart=shared 
./batch_graph  1
./batch_graph  2
./batch_graph  4
./batch_graph  8
./batch_graph 10
./batch_graph 100
./batch_graph 1000
./batch_graph 2000
./batch_graph 3000
./batch_graph 4000
./batch_graph 5000
./batch_graph 6000
./batch_graph 7000
./batch_graph 8000
./batch_graph 9000
./batch_graph 10000

# ./example 10000
# > ${GVIRTUS_HOME}/frontend.log 2>&1
# ./example

# nvcc cnn.cu -o example -L ${GVIRTUS_HOME}/lib/frontend -L ${GVIRTUS_HOME}/lib/ -lcuda --cudart=shared 
# ./example