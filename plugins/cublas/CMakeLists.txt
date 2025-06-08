project(gvirtus-plugin-cublas)
find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

resolve_cuda_library_version(cublas CUBLAS_VERSION)

gvirtus_add_backend(cublas ${CUBLAS_VERSION}
        backend/CublasHandler.cpp)

target_link_libraries(${PROJECT_NAME}
    CUDA::cublas
)

gvirtus_add_frontend(cublas ${CUBLAS_VERSION}
    frontend/Cublas_helper.cpp
    frontend/Cublas_level1.cpp
    frontend/Cublas_level2.cpp
    frontend/Cublas_level3.cpp
    frontend/CublasFrontend.cpp
)
