cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)

project(GVirtuS LANGUAGES CXX)

# ---------- 基础设置 ----------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)

# ===== LZ4 check & link =====
find_package(LZ4 QUIET)
if(LZ4_FOUND)
    link_libraries(LZ4::LZ4)
else()
    find_library(LZ4_LIB lz4 REQUIRED)
    link_libraries(${LZ4_LIB})
endif()

enable_testing()

# Set required CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

# GVIRTUS_HOME
if("$ENV{GVIRTUS_HOME}" STREQUAL "")
    message(STATUS "Setting GVIRTUS_HOME=$ENV{HOME}/GVirtuS")
    set(GVIRTUS_HOME "$ENV{HOME}/GVirtuS")
    set($ENV{GVIRTUS_HOME} "${GVIRTUS_HOME}")
else()
    message(STATUS "Using GVIRTUS_HOME=$ENV{GVIRTUS_HOME}")
    set(GVIRTUS_HOME "$ENV{GVIRTUS_HOME}")
endif()

include(cmake/GVirtuS.cmake)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann/json.hpp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann
    COMMAND ${CMAKE_COMMAND} -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann/json.hpp -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/download_json.cmake
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# ---------- 通用 RPATH（避免运行时混库） ----------
# 构建时：允许从构建目录和系统路径加载
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
# 安装后：优先 $ORIGIN（同目录）与 ${GVIRTUS_HOME}/lib
set(CMAKE_INSTALL_RPATH "\$ORIGIN;${GVIRTUS_HOME}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ===== gvirtus-common =====
add_library(gvirtus-common SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann/json.hpp
    src/common/Decoder.cpp
    src/common/Encoder.cpp
    src/common/JSON.cpp
    src/common/LD_Lib.cpp
    src/common/MessageDispatcher.cpp
    src/common/Mutex.cpp
    src/common/Observable.cpp
    src/common/Observer.cpp
    src/common/SignalException.cpp
    src/common/SignalState.cpp
    src/common/Util.cpp
)
target_link_libraries(gvirtus-common
    stdc++fs
    Threads::Threads
    ${CMAKE_DL_LIBS}
    ${LIBLOG4CPLUS}
    rdmacm
    ibverbs
)

gvirtus_install_target(gvirtus-common)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/nlohmann DESTINATION ${GVIRTUS_HOME}/include)
if (LOG4CPLUS_EXTERNALLY_DOWNLOADED)
    add_dependencies(gvirtus-common log4cplus)
    install(DIRECTORY ${EXTERNAL_INSTALL_LOCATION}/include/log4cplus DESTINATION ${GVIRTUS_HOME}/include)
    install(DIRECTORY ${EXTERNAL_INSTALL_LOCATION}/lib DESTINATION ${GVIRTUS_HOME})
endif()
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/etc DESTINATION ${GVIRTUS_HOME})

# ===== gvirtus-communicators =====
add_library(gvirtus-communicators SHARED
    src/communicators/Buffer.cpp
    src/communicators/CommunicatorFactory.cpp
    src/communicators/Endpoint_Tcp.cpp
    src/communicators/Endpoint_Rdma.cpp
    src/communicators/Endpoint_Hybrid.cpp
    src/communicators/Endpoint_Ucx.cpp
    src/communicators/EndpointFactory.cpp
    src/communicators/rdma/ktmrdma.cpp
    src/communicators/Result.cpp
    include/gvirtus/communicators/UcxProtocol.h
)
target_link_libraries(gvirtus-communicators gvirtus-common rdmacm ibverbs)
gvirtus_install_target(gvirtus-communicators)

## TCP COMMUNICATOR
add_library(gvirtus-communicators-tcp SHARED
    src/communicators/tcp/TcpCommunicator.cpp)
target_link_libraries(gvirtus-communicators-tcp gvirtus-communicators)
gvirtus_install_target(gvirtus-communicators-tcp)

## IB (RDMA) COMMUNICATOR
add_library(gvirtus-communicators-ib SHARED
    src/communicators/rdma/ktmrdma.cpp
    src/communicators/rdma/RdmaCommunicator.cpp
)
target_link_libraries(gvirtus-communicators-ib gvirtus-communicators rdmacm ibverbs)
gvirtus_install_target(gvirtus-communicators-ib)

## HYBRID COMMUNICATOR
add_library(gvirtus-communicators-hybrid SHARED
    src/communicators/hybrid/HybridCommunicator.cpp
)
target_include_directories(gvirtus-communicators-hybrid
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(gvirtus-communicators-hybrid
    gvirtus-communicators
    gvirtus-communicators-tcp
    gvirtus-communicators-ib
)
gvirtus_install_target(gvirtus-communicators-hybrid)

# ===== UCX COMMUNICATOR（与 RDMA 同等级：显式找全并链接 ucp/uct/ucs/ucm） =====

# 1) 优先使用 pkg-config
if (PkgConfig_FOUND)
    pkg_check_modules(UCX QUIET ucx)
endif()

# 2) 兜底：手动找头文件与四个库
if (NOT UCX_FOUND)
    find_path(UCX_INCLUDE_DIR ucp/api/ucp.h)
    find_library(UCX_UCP_LIBRARY NAMES ucp)
    find_library(UCX_UCT_LIBRARY NAMES uct)
    find_library(UCX_UCS_LIBRARY NAMES ucs)
    find_library(UCX_UCM_LIBRARY NAMES ucm) # 一般可选，但这里显式要求，避免运行时拉错版本

    if (UCX_INCLUDE_DIR AND UCX_UCP_LIBRARY AND UCX_UCT_LIBRARY AND UCX_UCS_LIBRARY AND UCX_UCM_LIBRARY)
        set(UCX_FOUND TRUE)
        set(UCX_INCLUDE_DIRS ${UCX_INCLUDE_DIR})
        set(UCX_LIBRARIES ${UCX_UCP_LIBRARY} ${UCX_UCT_LIBRARY} ${UCX_UCS_LIBRARY} ${UCX_UCM_LIBRARY})
    endif()
else()
    # 使用 pkg-config 的结果时，仍然显式分解四个库，确保链接完整
    find_library(UCX_UCP_LIBRARY NAMES ucp)
    find_library(UCX_UCT_LIBRARY NAMES uct)
    find_library(UCX_UCS_LIBRARY NAMES ucs)
    find_library(UCX_UCM_LIBRARY NAMES ucm)

    if (NOT (UCX_UCP_LIBRARY AND UCX_UCT_LIBRARY AND UCX_UCS_LIBRARY AND UCX_UCM_LIBRARY))
        message(FATAL_ERROR "Found UCX via pkg-config, but failed to locate all required libs (ucp/uct/ucs/ucm).")
    endif()

    set(UCX_LIBRARIES ${UCX_UCP_LIBRARY} ${UCX_UCT_LIBRARY} ${UCX_UCS_LIBRARY} ${UCX_UCM_LIBRARY})
endif()

if (NOT UCX_FOUND)
    message(FATAL_ERROR "UCX not found: please install libucx-dev / ucx-devel or provide -DUCX_INCLUDE_DIR / -DUCX_*_LIBRARY")
endif()

# 3) 打印并校验四个 UCX 库来自同一目录（防止混库）
foreach(_lib ${UCX_LIBRARIES})
    get_filename_component(_dir "${_lib}" DIRECTORY)
    list(APPEND _UCX_DIRS "${_dir}")
endforeach()
list(REMOVE_DUPLICATES _UCX_DIRS)

message(STATUS "UCX include dirs: ${UCX_INCLUDE_DIRS}")
message(STATUS "UCX libraries   : ${UCX_LIBRARIES}")
message(STATUS "UCX library dirs: ${_UCX_DIRS}")

list(LENGTH _UCX_DIRS _UCX_DIRS_LEN)
if (_UCX_DIRS_LEN GREATER 1)
    message(FATAL_ERROR "UCX libs are from multiple directories (${_UCX_DIRS}). This is likely to cause runtime symbol errors. Ensure all ucp/uct/ucs/ucm come from the same installation.")
endif()

link_directories(${_UCX_DIRS})

# 4) 生成 UCX communicator
add_library(gvirtus-communicators-ucx SHARED
    src/communicators/ucx/UcxCommunicator.cpp
)
target_include_directories(gvirtus-communicators-ucx
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${UCX_INCLUDE_DIRS}
)
target_link_libraries(gvirtus-communicators-ucx
    PRIVATE
        gvirtus-communicators
        ${UCX_LIBRARIES}      # 与 RDMA 一致：显式列出依赖
)
# RPATH：让插件加载到与其同目录的 UCX
set_target_properties(gvirtus-communicators-ucx PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN;${GVIRTUS_HOME}/lib"
)
gvirtus_install_target(gvirtus-communicators-ucx)

# 安装协议头
install(FILES include/gvirtus/communicators/UcxProtocol.h
        DESTINATION ${GVIRTUS_HOME}/include/gvirtus/communicators)

# ===== FRONTEND =====
add_library(gvirtus-frontend SHARED
    src/frontend/Frontend.cpp
)
target_include_directories(gvirtus-frontend
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(gvirtus-frontend
    PRIVATE
        gvirtus-communicators
        gvirtus-communicators-hybrid
        gvirtus-communicators-ucx
        Threads::Threads
        lz4
        ${UCX_LIBRARIES}
)
set_target_properties(gvirtus-frontend PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN;${GVIRTUS_HOME}/lib"
)
gvirtus_install_target(gvirtus-frontend)

# ===== BACKEND =====
add_executable(gvirtus-backend
    src/backend/Backend.cpp
    src/backend/main.cpp
    src/backend/Process.cpp
    src/backend/Property.cpp
)
target_include_directories(gvirtus-backend
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(gvirtus-backend
    PRIVATE
        gvirtus-communicators
        gvirtus-communicators-hybrid
        gvirtus-communicators-ucx
        Threads::Threads
        rdmacm
        ibverbs
        ${UCX_LIBRARIES}   # <--- 加上这一行
)
set_target_properties(gvirtus-backend PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN;${GVIRTUS_HOME}/lib"
)
gvirtus_install_target(gvirtus-backend)

# ===== Plugins =====
add_subdirectory(plugins/cudart)
add_subdirectory(plugins/cudadr)
add_subdirectory(plugins/cudnn)

maybe_add_cuda_plugin(cublas plugins/cublas)
maybe_add_cuda_plugin(cufft plugins/cufft)
maybe_add_cuda_plugin(cusolver plugins/cusolver)
maybe_add_cuda_plugin(cusparse plugins/cusparse)
maybe_add_cuda_plugin(curand plugins/curand)
maybe_add_cuda_plugin(nvrtc plugins/nvrtc)

add_subdirectory(plugins/nvml)

# ===== Tests =====
add_subdirectory(tests)

# add_subdirectory(tools/protocol-generator)
