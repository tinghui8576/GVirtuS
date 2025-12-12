FROM tinghui8576/gvirtus_backend_run:latest

WORKDIR /gvirtus/example
COPY example.cu .

ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

RUN nvcc -o example example.cu -L ${GVIRTUS_HOME}/lib/frontend -lcublas -lcudnn -lcurand -lcufft --cudart=shared

ENTRYPOINT ["/bin/bash"]
