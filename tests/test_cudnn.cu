#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess)
#define CUDNN_CHECK(err) ASSERT_EQ((err), CUDNN_STATUS_SUCCESS)

TEST(cuDNN, CreateDestroy) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, GetVersion) {
    size_t version = cudnnGetVersion();
    ASSERT_GT(version, 0);
    std::cout << "cuDNN version: " << version << std::endl;
}

TEST(cuDNN, SetGetStream) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDNN_CHECK(cudnnSetStream(handle, stream));

    cudaStream_t got_stream;
    CUDNN_CHECK(cudnnGetStream(handle, &got_stream));
    ASSERT_EQ(stream, got_stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, TensorDescriptorCreateDestroy) {
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
}

TEST(cuDNN, TensorDescriptorSetGet) {
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 3, 4));

    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;

    CUDNN_CHECK(cudnnGetTensor4dDescriptor(desc, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
    ASSERT_EQ(n, 1);
    ASSERT_EQ(c, 2);
    ASSERT_EQ(h, 3);
    ASSERT_EQ(w, 4);

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
}

TEST(cuDNN, ActivationForwardReLU) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 3));

    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    float h_input[] = {-1.0f, 0.0f, 2.0f};
    float h_output[3] = {};

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(h_input)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(h_output)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnActivationForward(handle, actDesc, &alpha, desc, d_input, &beta, desc, d_output));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    ASSERT_FLOAT_EQ(h_output[0], 0.0f);
    ASSERT_FLOAT_EQ(h_output[1], 0.0f);
    ASSERT_FLOAT_EQ(h_output[2], 2.0f);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, PoolingForwardMax) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnPoolingDescriptor_t pool_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 2, 2));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            2, 2, 0, 0, 1, 1));

    float h_input[] = {1, 2, 3, 4}; // max is 4
    float h_output[1] = {};

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(h_input)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(h_output)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(handle, pool_desc, &alpha, in_desc, d_input, &beta, out_desc, d_output));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    ASSERT_FLOAT_EQ(h_output[0], 4.0f);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, FilterDescriptorCreateDestroy) {
    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
}

TEST(cuDNN, ErrorString) {
    const char* msg = cudnnGetErrorString(CUDNN_STATUS_ALLOC_FAILED);
    ASSERT_TRUE(msg != nullptr);
}
