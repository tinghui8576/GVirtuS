FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    libxmu-dev \
    libxi-dev \
    libgl-dev \
    libosmesa-dev \
    liblog4cplus-dev \
    librdmacm-dev \
    libibverbs-dev \
    autotools-dev \
    automake \
    cmake \
    pkg-config \
    curl \
    git \
    # gcc-10 \
    # g++-10 \
    && rm -rf /var/lib/apt/lists/*

# COPY cmake /gvirtus/cmake
# COPY etc /gvirtus/etc
# COPY include /gvirtus/include
# COPY plugins /gvirtus/plugins
# COPY src /gvirtus/src
# COPY tools /gvirtus/tools
# COPY CMakeLists.txt /gvirtus/CMakeLists.txt

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# ENV GVIRTUS_HOME=/usr/local/gvirtus
# RUN mkdir gvirtus/build && cd gvirtus/build && cmake .. && make && make install
# ENV GVIRTUS_LOGLEVEL=0
# ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}
# RUN sed -i 's/"server_address": "127.0.0.1"/"server_address": "0.0.0.0"/' ${GVIRTUS_HOME}/etc/properties.json
# ENTRYPOINT ["/bin/bash", "-c", "${GVIRTUS_HOME}/bin/gvirtus-backend ${GVIRTUS_HOME}/etc/properties.json"]