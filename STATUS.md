# cudaDr (-lcuda)

| Function                          | Implemented | Tested  | Working |        Notes             |
| --------------------------------- | ----------- | ------- | ------- | ------------------------ |
| `cuInit`                          | ✅          | ❌      | ❓      |                          |
| `cuCtxCreate`                     | ✅          | ❌      | ❓      |                          |
| `cuCtxAttach`                     | ✅          | ❌      | ❓      | Deprecated               |
| `cuCtxDestroy`                    | ✅          | ❌      | ❓      |                          |
| `cuCtxDetach`                     | ✅          | ❌      | ❓      | Deprecated               |
| `cuCtxSetCurrent`                 | ❌          | ❌      | ❌      |                          |
| `cuCtxGetCurrent`                 | ❌          | ❌      | ❌      |                          |
| `cuCtxGetDevice`                  | ✅          | ❌      | ❓      |                          |
| `cuCtxPopCurrent`                 | ✅          | ❌      | ❓      |                          |
| `cuCtxPushCurrent`                | ✅          | ❌      | ❓      |                          |
| `cuCtxSynchronize`                | ✅          | ❌      | ❓      |                          |
| `cuCtxDisablePeerAccess`          | ✅          | ❌      | ❓      |                          |
| `cuCtxEnablePeerAccess`           | ✅          | ❌      | ❓      |                          |
| `cuDeviceCanAccessPeer`           | ✅          | ❌      | ❓      |                          |
| `cuDeviceComputeCapability`       | ✅          | ❌      | ❓      | Deprecated               |
| `cuDeviceGet`                     | ✅          | ❌      | ❓      |                          |
| `cuDeviceGetAttribute`            | ✅          | ❌      | ❓      |                          |
| `cuDeviceGetCount`                | ✅          | ❌      | ❓      |                          |
| `cuDeviceGetName`                 | ✅          | ❌      | ❓      |                          |
| `cuDeviceGetProperties`           | ✅          | ❌      | ❓      | Deprecated               |
| `cuDeviceTotalMem`                | ✅          | ❌      | ❓      |                          |
| `cuParamSetSize`                  | ✅          | ❌      | ❓      | Deprecated               |
| `cuFuncSetBlockShape`             | ✅          | ❌      | ❓      | Deprecated               |
| `cuLaunchGrid`                    | ✅          | ❌      | ❓      | Deprecated               |
| `cuFuncGetAttribute`              | ✅          | ❌      | ❓      |                          |
| `cuFuncSetSharedSize`             | ✅          | ❌      | ❓      | Deprecated               |
| `cuLaunch`                        | ✅          | ❌      | ❓      | Deprecated               |
| `cuParamSetf`                     | ✅          | ❌      | ❓      | Deprecated               |
| `cuParamSeti`                     | ✅          | ❌      | ❓      | Deprecated               |
| `cuParamSetv`                     | ✅          | ❌      | ❓      | Deprecated               |
| `cuParamSetTexRef`                | ✅          | ❌      | ❓      | Deprecated               |
| `cuLaunchGridAsync`               | ✅          | ❌      | ❓      | Deprecated               |
| `cuFuncSetCacheConfig`            | ✅          | ❌      | ❓      |                          |
| `cuMemFree`                       | ✅          | ❌      | ❓      |                          |
| `cuMemAlloc`                      | ✅          | ❌      | ❓      |                          |
| `cuMemAllocManaged`               | ❌          | ❌      | ❌      |                          |
| `cuMemHostAlloc`                  | ❌          | ❌      | ❌      |                          |
| `cuMemHostFree`                   | ❌          | ❌      | ❌      |                          |
| `cuMemcpyDtoH`                    | ✅          | ❌      | ❓      |                          |
| `cuMemcpyHtoD`                    | ✅          | ❌      | ❓      |                          |
| `cuMemcpyDtoD`                    | ❌          | ❌      | ❌      |                          |
| `cuMemcpyHtoDAsync`               | ❌          | ❌      | ❌      |                          |
| `cuMemcpyDtoHAsync`               | ❌          | ❌      | ❌      |                          |
| `cuMemsetD32`                     | ❌          | ❌      | ❌      |                          |
| `cuMemsetD8`                      | ❌          | ❌      | ❌      |                          |
| `cuArrayCreate`                   | ✅          | ❌      | ❓      |                          |
| `cuMemcpy2D`                      | ✅          | ❌      | ❓      |                          |
| `cuArrayDestroy`                  | ✅          | ❌      | ❓      |                          |
| `cuArray3DCreate`                 | ✅          | ❌      | ❓      |                          |
| `cuMemAllocPitch`                 | ✅          | ❌      | ❓      |                          |
| `cuMemGetAddressRange`            | ✅          | ❌      | ❓      |                          |
| `cuMemGetInfo`                    | ✅          | ❌      | ❓      |                          |
| `cuModuleLoadData`                | ✅          | ❌      | ❓      |                          |
| `cuModuleLoad`                    | ✅          | ❌      | ❓      |                          |
| `cuModuleLoadFatBinary`           | ✅          | ❌      | ❓      |                          |
| `cuModuleUnload`                  | ✅          | ❌      | ❓      |                          |
| `cuModuleGetFunction`             | ✅          | ❌      | ❓      |                          |
| `cuModuleGetGlobal`               | ✅          | ❌      | ❓      |                          |
| `cuModuleLoadDataEx`              | ✅          | ❌      | ❓      |                          |
| `cuModuleGetTexRef`               | ✅          | ❌      | ❓      | Deprecated               |
| `cuDriverGetVersion`              | ✅          | ❌      | ❓      |                          |
| `cuStreamCreate`                  | ✅          | ❌      | ❓      |                          |
| `cuStreamDestroy`                 | ✅          | ❌      | ❓      |                          |
| `cuStreamQuery`                   | ✅          | ❌      | ❓      |                          |
| `cuStreamSynchronize`             | ✅          | ❌      | ❓      |                          |
| `cuEventCreate`                   | ✅          | ❌      | ❓      |                          |
| `cuEventDestroy`                  | ✅          | ❌      | ❓      |                          |
| `cuEventElapsedTime`              | ✅          | ❌      | ❓      |                          |
| `cuEventQuery`                    | ✅          | ❌      | ❓      |                          |
| `cuEventRecord`                   | ✅          | ❌      | ❓      |                          |
| `cuEventSynchronize`              | ✅          | ❌      | ❓      |                          |
| `cuLinkCreate`                    | ❌          | ❌      | ❌      |                          |
| `cuLinkAddData`                   | ❌          | ❌      | ❌      |                          |
| `cuLinkComplete`                  | ❌          | ❌      | ❌      |                          |
| `cuModuleLoadDataEx`              | ❌          | ❌      | ❌      |                          |
| `cuGraphicsGLRegisterBuffer`      | ❌          | ❌      | ❌      |                          |
| `cuGraphicsMapResources`          | ❌          | ❌      | ❌      |                          |
| `cuTexRefSetArray`                | ✅          | ❌      | ❓      | Deprecated               |
| `cuTexRefSetAddressMode`          | ✅          | ❌      | ❓      | Deprecated               |
| `cuTexRefSetFilterMode`           | ✅          | ❌      | ❓      | Deprecated               |
| `cuTexRefSetFlags`                | ✅          | ❌      | ❓      | Deprecated               |
| `cuTexRefSetFormat`               | ✅          | ❌      | ❓      | Deprecated               |
| `cuTexRefGetAddress`              | ✅          | ❌      | ❓      | Deprecated               |
| `cuTexRefGetArray`                | ✅          | ❌      | ❓      | Deprecated               |
| `cuTexRefGetFlags`                | ✅          | ❌      | ❓      | Deprecated               |
| `cuTexRefSetAddress`              | ✅          | ❌      | ❓      | Deprecated               |
| `cuLaunchKernel`                  | ✅          | ❌      | ❓      |                          |

# cudaRT (-lcudart)

| Function                                                 | Implemented | Tested | Working  |          Notes          |
| -------------------------------------------------------- | ----------- | ------ | -------- | ----------------------- |
| `cudaMalloc`                                             | ✅          | ✅      | ✅      |                         |
| `cudaFree`                                               | ✅          | ✅      | ✅      |                         |
| `cudaMallocHost`                                         | ❌          | ❌      | ❌      |                         |
| `cudaFreeHost`                                           | ❌          | ❌      | ❌      |                         |
| `cudaMemcpy`                                             | ✅          | ✅      | ✅      |                         |
| `cudaMemcpyAsync`                                        | ✅          | ✅      | ✅      |                         |
| `cudaMemset`                                             | ✅          | ✅      | ✅      |                         |
| `cudaMemsetAsync`                                        | ✅          | ✅      | ✅      |                         |
| `cudaGetDevice`                                          | ✅          | ✅      | ✅      |                         |
| `cudaSetDevice`                                          | ✅          | ✅      | ✅      |                         |
| `cudaStreamCreate`                                       | ✅          | ✅      | ✅      |                         |
| `cudaStreamQuery`                                        | ✅          | ❌      | ❓      |                         |
| `cudaStreamSynchronize`                                  | ✅          | ❌      | ❓      |                         |
| `cudaStreamCreateWithFlags`                              | ✅          | ❌      | ❓      |                         |
| `cudaStreamWaitEvent`                                    | ✅          | ❌      | ❓      |                         |
| `cudaStreamCreateWithPriority`                           | ✅          | ❌      | ❓      |                         |
| `cudaStreamDestroy`                                      | ✅          | ✅      | ✅      |                         |
| `cudaEventCreate`                                        | ✅          | ✅      | ✅      |                         |
| `cudaEventCreateWithFlags`                               | ✅          | ❌      | ❓      |                         |
| `cudaEventQuery`                                         | ✅          | ❌      | ❓      |                         |
| `cudaEventRecord`                                        | ✅          | ✅      | ✅      |                         |
| `cudaEventSynchronize`                                   | ✅          | ✅      | ✅      |                         |
| `cudaEventElapsedTime`                                   | ✅          | ✅      | ✅      |                         |
| `cudaEventDestroy`                                       | ✅          | ✅      | ✅      |                         |
| `cudaChooseDevice`                                       | ✅          | ❌      | ❓      |                         |
| `cudaGetDeviceCount`                                     | ✅          | ❌      | ❓      |                         |
| `cudaGetDeviceProperties`                                | ✅          | ❌      | ❓      |                         |
| `cudaSetDeviceFlags`                                     | ✅          | ❌      | ❓      |                         |
| `cudaSetValidDevices`                                    | ✅          | ❌      | ❓      |                         |
| `cudaDeviceReset`                                        | ✅          | ❌      | ❓      |                         |
| `cudaDeviceSynchronize`                                  | ✅          | ✅      | ✅      |                         |
| `cudaDeviceSetCacheConfig`                               | ✅          | ❌      | ❓      |                         |
| `cudaDeviceSetLimit`                                     | ✅          | ❌      | ❓      |                         |
| `cudaDeviceCanAccessPeer`                                | ✅          | ❌      | ❓      |                         |
| `cudaDeviceEnablePeerAccess`                             | ✅          | ❌      | ❓      |                         |
| `cudaDeviceDisablePeerAccess`                            | ✅          | ❌      | ❓      |                         |
| `cudaIpcGetMemHandle`                                    | ✅          | ❌      | ❓      |                         |
| `cudaIpcGetEventHandle`                                  | ✅          | ❌      | ❓      |                         |
| `cudaIpcOpenEventHandle`                                 | ✅          | ❌      | ❓      |                         |
| `cudaIpcOpenMemHandle`                                   | ✅          | ❌      | ❓      |                         |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`          | ✅          | ❌      | ❓      |                         |
| `cudaDeviceGetAttribute`                                 | ✅          | ❌      | ❓      |                         |
| `cudaDeviceGetStreamPriorityRange`                       | ✅          | ❌      | ❓      |                         |
| `cudaGetErrorString`                                     | ✅          | ❌      | ❓      |                         |
| `cudaGetLastError`                                       | ✅          | ❌      | ❓      |                         |
| `cudaPeekAtLastError`                                    | ✅          | ❌      | ❓      |                         |
| `cudaFuncGetAttributes`                                  | ✅          | ❌      | ❓      |                         |
| `cudaFuncSetCacheConfig`                                 | ✅          | ❌      | ❓      |                         |
| `cudaLaunchKernel`                                       | ✅          | ✅      | ✅      | Tested using both `<<<>>>` and direct syntax |
| `__cudaPushCallConfiguration`                            | ✅          | ✅      | ✅      | Tested implicitly using `<<<>>>` syntax |
| `__cudaPopCallConfiguration`                             | ✅          | ✅      | ✅      | Tested implicitly using `<<<>>>` syntax |
| `cudaLaunch`                                             | ✅          | ❌      | ❓      | This function is deprecated as of CUDA 7.0 |
| `cudaConfigureCall`                                      | ✅          | ❌      | ❓      | This function is deprecated as of CUDA 7.0 |
| `cudaSetupArgument`                                      | ✅          | ❌      | ❓      | This function is deprecated as of CUDA 7.0  |
| `cudaRegisterFatBinary`                                  | ✅          | ✅      | ✅      | Automatically called at program start |
| `cudaRegisterFatBinaryEnd`                               | ✅          | ✅      | ✅      | Automatically called after `cudaRegisterFatBinary` |
| `cudaUnregisterFatBinary`                                | ✅          | ✅      | ✅      | Automatically called at program exit |
| `cudaSetDoubleForHost`                                   | ✅          | ❌      | ❓      |                         |
| `cudaSetDoubleForDevice`                                 | ✅          | ❌      | ❓      |                         |
| `cudaRegisterFunction`                                   | ✅          | ❌      | ❓      |                         |
| `cudaRegisterVar`                                        | ✅          | ❌      | ❓      |                         |
| `cudaRegisterSharedVar`                                  | ✅          | ❌      | ❓      |                         |
| `cudaRegisterShared`                                     | ✅          | ❌      | ❓      |                         |
| `cudaRegisterTexture`                                    | ✅          | ❌      | ❓      |                         |
| `cudaRegisterSurface`                                    | ✅          | ❌      | ❓      |                         |
| `cudaRegisterSharedMemory`                               | ✅          | ❌      | ❓      |                         |
| `cudaRequestSharedMemory`                                | ✅          | ❌      | ❓      |                         |
| `cudaFreeArray`                                          | ✅          | ❌      | ❓      |                         |
| `cudaGetSymbolAddress`                                   | ✅          | ❌      | ❓      |                         |
| `cudaGetSymbolSize`                                      | ✅          | ❌      | ❓      |                         |
| `cudaMallocArray`                                        | ✅          | ❌      | ❓      |                         |
| `cudaMallocPitch`                                        | ✅          | ❌      | ❓      |                         |
| `cudaMallocManaged`                                      | ✅          | ❌      | ❓      |                         |
| `cudaMemcpy2D`                                           | ✅          | ❌      | ❓      |                         |
| `cudaMemcpy3D`                                           | ✅          | ❌      | ❓      |                         |
| `cudaMemcpyFromSymbol`                                   | ✅          | ❌      | ❓      |                         |
| `cudaMemcpyToArray`                                      | ✅          | ❌      | ❓      | Deprecated              |
| `cudaMemcpyToSymbol`                                     | ✅          | ❌      | ❓      |                         |
| `cudaMemset2D`                                           | ✅          | ❌      | ❓      |                         |
| `cudaMemcpyFromArray`                                    | ✅          | ❌      | ❓      | Deprecated              |
| `cudaMemcpyArrayToArray`                                 | ✅          | ❌      | ❓      | Deprecated              |
| `cudaMemcpy2DFromArray`                                  | ✅          | ❌      | ❓      |                         |
| `cudaMemcpy2DToArray`                                    | ✅          | ❌      | ❓      |                         |
| `cudaMalloc3DArray`                                      | ✅          | ❌      | ❓      |                         |
| `cudaMemcpyPeerAsync`                                    | ✅          | ❌      | ❓      |                         |
| `cudaGLSetGLDevice`                                      | ✅          | ❌      | ❓      | Deprecated              |
| `cudaGraphicsGLRegisterBuffer`                           | ✅          | ❌      | ❓      |                         |
| `cudaGraphicsMapResources`                               | ✅          | ❌      | ❓      |                         |
| `cudaGraphicsResourceGetMappedPointer`                   | ✅          | ❌      | ❓      |                         |
| `cudaGraphicsUnmapResources`                             | ✅          | ❌      | ❓      |                         |
| `cudaGraphicsUnregisterResource`                         | ✅          | ❌      | ❓      |                         |
| `cudaGraphicsResourceSetMapFlags`                        | ✅          | ❌      | ❓      |                         |
| `cudaBindTexture`                                        | ✅          | ❌      | ❓      | Deprecated              |
| `cudaBindTexture2D`                                      | ✅          | ❌      | ❓      | Deprecated              |
| `cudaBindTextureToArray`                                 | ✅          | ❌      | ❓      | Deprecated              |
| `cudaCreateTextureObject`                                | ✅          | ❌      | ❓      |                         |
| `cudaGetChannelDesc`                                     | ✅          | ❌      | ❓      |                         |
| `cudaGetTextureAlignmentOffset`                          | ✅          | ❌      | ❓      | Deprecated              |
| `cudaGetTextureReference`                                | ✅          | ❌      | ❓      | Deprecated              |
| `cudaUnbindTexture`                                      | ✅          | ❌      | ❓      | Deprecated              |
| `cudaBindSurfaceToArray`                                 | ✅          | ❌      | ❓      |                         |
| `cudaGetTextureReference`                                | ✅          | ❌      | ❓      |                         |
| `cudaThreadExit`                                         | ✅          | ❌      | ❓      | Deprecated in favor of `cudaDeviceReset` |
| `cudaThreadSynchronize`                                  | ✅          | ❌      | ❓      | Deprecated in favor of `cudaDeviceSynchronize` |
| `cudaDriverGetVersion`                                   | ✅          | ❌      | ❓      |                         |
| `cudaRuntimeGetVersion`                                  | ✅          | ❌      | ❓      |                         |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor`          | ✅          | ❌      | ❓      |                         |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | ✅          | ❌      | ❓      |                         |


# cuBLAS (-lcublas)

| Function                | Implemented | Tested | Working  |          Notes          |
| ----------------------- | ----------- | ------ | -------- | ------------------------|
| `cublasCreate`          | ✅          | ✅      | ✅      |                         |
| `cublasDestroy`         | ✅          | ✅      | ✅      |                         |
| `cublasGetVersion`      | ✅          | ✅      | ✅      |                         |
| `cublasSetStream`       | ✅          | ✅      | ❌      |     broke in CUDA 12    |
| `cublasSetVector`       | ✅          | ❌      | ❓      |                         |
| `cublasGetVector`       | ✅          | ❌      | ❓      |                         |
| `cublasSetMatrix`       | ✅          | ❌      | ❓      |                         |
| `cublasGetMatrix`       | ✅          | ❌      | ❓      |                         |
| `cublasGetPointerMode`  | ✅          | ❌      | ❓      |                         |
| `cublasSetPointerMode`  | ✅          | ❌      | ❓      |                         |
| `cublasSaxpy`           | ✅          | ✅      | ✅      |                         |
| `cublasDaxpy`           | ✅          | ✅      | ❌      |     broke in CUDA 12    |
| `cublasCaxpy`           | ✅          | ❌      | ❓      |                         |
| `cublasZaxpy`           | ✅          | ❌      | ❓      |                         |
| `cublasScopy`           | ✅          | ✅      | ✅      |                         |
| `cublasDcopy`           | ✅          | ✅      | ✅      |                         |
| `cublasCcopy`           | ✅          | ❌      | ❓      |                         |
| `cublasZcopy`           | ✅          | ❌      | ❓      |                         |
| `cublasSnrm2`           | ✅          | ✅      | ❌      |     broke in CUDA 12    |
| `cublasDnrm2`           | ✅          | ✅      | ❌      |     broke in CUDA 12    |
| `cublasSgemm`           | ✅          | ✅      | ❌      |                         |
| `cublasDgemm`           | ✅          | ✅      | ❌      |                         |
| `cublasSgemv`           | ✅          | ✅      | ❌      |                         |
| `cublasDgemv`           | ✅          | ✅      | ❌      |                         |
| `cublasSdot`            | ✅          | ✅      | ❌      |                         |
| `cublasDdot`            | ✅          | ✅      | ❌      |                         |
| `cublasCdotu`           | ✅          | ❌      | ❓      |                         |
| `cublasCdotc`           | ✅          | ❌      | ❓      |                         |
| `cublasZdotu`           | ✅          | ❌      | ❓      |                         |
| `cublasZdotc`           | ✅          | ❌      | ❓      |                         |
| `cublasSscal`           | ✅          | ❌      | ❓      |                         |
| `cublasDscal`           | ✅          | ❌      | ❓      |                         |
| `cublasCscal`           | ✅          | ❌      | ❓      |                         |
| `cublasCsscal`          | ✅          | ❌      | ❓      |                         |
| `cublasZscal`           | ✅          | ❌      | ❓      |                         |
| `cublasZdscal`          | ✅          | ❌      | ❓      |                         |
| `cublasSswap`           | ✅          | ❌      | ❓      |                         |
| `cublasDswap`           | ✅          | ❌      | ❓      |                         |
| `cublasCswap`           | ✅          | ❌      | ❓      |                         |
| `cublasZswap`           | ✅          | ❌      | ❓      |                         |
| `cublasIsamax`          | ✅          | ❌      | ❓      |                         |
| `cublasIdamax`          | ✅          | ❌      | ❓      |                         |
| `cublasIcamax`          | ✅          | ❌      | ❓      |                         |
| `cublasIzamax`          | ✅          | ❌      | ❓      |                         |
| `cublasSasum`           | ✅          | ❌      | ❓      |                         |
| `cublasDasum`           | ✅          | ❌      | ❓      |                         |
| `cublasScasum`          | ✅          | ❌      | ❓      |                         |
| `cublasDzasum`          | ✅          | ❌      | ❓      |                         |
| `cublasSrot`            | ✅          | ❌      | ❓      |                         |
| `cublasDrot`            | ✅          | ❌      | ❓      |                         |
| `cublasCrot`            | ✅          | ❌      | ❓      |                         |
| `cublasCsrot`           | ✅          | ❌      | ❓      |                         |
| `cublasZrot`            | ✅          | ❌      | ❓      |                         |
| `cublasZdrot`           | ✅          | ❌      | ❓      |                         |
| `cublasSrotg`           | ✅          | ❌      | ❓      |                         |
| `cublasDrotg`           | ✅          | ❌      | ❓      |                         |
| `cublasCrotg`           | ✅          | ❌      | ❓      |                         |
| `cublasZrotg`           | ✅          | ❌      | ❓      |                         |
| `cublasSrotm`           | ✅          | ❌      | ❓      |                         |
| `cublasDrotm`           | ✅          | ❌      | ❓      |                         |
| `cublasSrotmg`          | ✅          | ❌      | ❓      |                         |
| `cublasDrotmg`          | ✅          | ❌      | ❓      |                         |
| `cublasSgemv`           | ✅          | ❌      | ❓      |                         |
| `cublasDgemv`           | ✅          | ❌      | ❓      |                         |
| `cublasCgemv`           | ✅          | ❌      | ❓      |                         |
| `cublasZgemv`           | ✅          | ❌      | ❓      |                         |
| `cublasSgbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasDgbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasSgbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasSgbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasStrmv`           | ✅          | ❌      | ❓      |                         |
| `cublasDtrmv`           | ✅          | ❌      | ❓      |                         |
| `cublasCtrmv`           | ✅          | ❌      | ❓      |                         |
| `cublasZtrmv`           | ✅          | ❌      | ❓      |                         |
| `cublasStbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasDtbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasCtbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasZtbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasStpmv`           | ✅          | ❌      | ❓      |                         |
| `cublasDtpmv`           | ✅          | ❌      | ❓      |                         |
| `cublasCtpmv`           | ✅          | ❌      | ❓      |                         |
| `cublasZtpmv`           | ✅          | ❌      | ❓      |                         |
| `cublasStpsv`           | ✅          | ❌      | ❓      |                         |
| `cublasDtpsv`           | ✅          | ❌      | ❓      |                         |
| `cublasCtpsv`           | ✅          | ❌      | ❓      |                         |
| `cublasZtpsv`           | ✅          | ❌      | ❓      |                         |
| `cublasStbsv`           | ✅          | ❌      | ❓      |                         |
| `cublasDtbsv`           | ✅          | ❌      | ❓      |                         |
| `cublasCtbsv`           | ✅          | ❌      | ❓      |                         |
| `cublasZtbsv`           | ✅          | ❌      | ❓      |                         |
| `cublasSsymv`           | ✅          | ❌      | ❓      |                         |
| `cublasDsymv`           | ✅          | ❌      | ❓      |                         |
| `cublasCsymv`           | ✅          | ❌      | ❓      |                         |
| `cublasZsymv`           | ✅          | ❌      | ❓      |                         |
| `cublasChemv`           | ✅          | ❌      | ❓      |                         |
| `cublasZhemv`           | ✅          | ❌      | ❓      |                         |
| `cublasSsbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasDsbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasChbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasZhbmv`           | ✅          | ❌      | ❓      |                         |
| `cublasSspmv`           | ✅          | ❌      | ❓      |                         |
| `cublasDspmv`           | ✅          | ❌      | ❓      |                         |
| `cublasChpmv`           | ✅          | ❌      | ❓      |                         |
| `cublasZhpmv`           | ✅          | ❌      | ❓      |                         |
| `cublasSger`            | ✅          | ❌      | ❓      |                         |
| `cublasDger`            | ✅          | ❌      | ❓      |                         |
| `cublasCgeru`           | ✅          | ❌      | ❓      |                         |
| `cublasCgerc`           | ✅          | ❌      | ❓      |                         |
| `cublasZgeru`           | ✅          | ❌      | ❓      |                         |
| `cublasZgerc`           | ✅          | ❌      | ❓      |                         |
| `cublasSsyr`            | ✅          | ❌      | ❓      |                         |
| `cublasDsyr`            | ✅          | ❌      | ❓      |                         |
| `cublasCsyr`            | ✅          | ❌      | ❓      |                         |
| `cublasZsyr`            | ✅          | ❌      | ❓      |                         |
| `cublasCher`            | ✅          | ❌      | ❓      |                         |
| `cublasZher`            | ✅          | ❌      | ❓      |                         |
| `cublasSspr`            | ✅          | ❌      | ❓      |                         |
| `cublasDspr`            | ✅          | ❌      | ❓      |                         |
| `cublasChpr`            | ✅          | ❌      | ❓      |                         |
| `cublasZhpr`            | ✅          | ❌      | ❓      |                         |
| `cublasSsyr2`           | ✅          | ❌      | ❓      |                         |
| `cublasDsyr2`           | ✅          | ❌      | ❓      |                         |
| `cublasCsyr2`           | ✅          | ❌      | ❓      |                         |
| `cublasZsyr2`           | ✅          | ❌      | ❓      |                         |
| `cublasCher2`           | ✅          | ❌      | ❓      |                         |
| `cublasZher2`           | ✅          | ❌      | ❓      |                         |
| `cublasSspr2`           | ✅          | ❌      | ❓      |                         |
| `cublasDspr2`           | ✅          | ❌      | ❓      |                         |
| `cublasChpr2`           | ✅          | ❌      | ❓      |                         |
| `cublasZhpr2`           | ✅          | ❌      | ❓      |                         |
| `cublasSgemm`           | ✅          | ❌      | ❓      |                         |
| `cublasDgemm`           | ✅          | ❌      | ❓      |                         |
| `cublasCgemm`           | ✅          | ❌      | ❓      |                         |
| `cublasZgemm`           | ✅          | ❌      | ❓      |                         |
| `cublasSgemmBatched`    | ✅          | ❌      | ❓      |                         |
| `cublasDgemmBatched`    | ✅          | ❌      | ❓      |                         |
| `cublasCgemmBatched`    | ✅          | ❌      | ❓      |                         |
| `cublasZgemmBatched`    | ✅          | ❌      | ❓      |                         |
| `cublasSnrm2`           | ✅          | ❌      | ❓      |                         |
| `cublasDnrm2`           | ✅          | ❌      | ❓      |                         |
| `cublasScnrm2`          | ✅          | ❌      | ❓      |                         |
| `cublasDznrm2`          | ✅          | ❌      | ❓      |                         |
| `cublasSsyrk`           | ✅          | ❌      | ❓      |                         |
| `cublasDsyrk`           | ✅          | ❌      | ❓      |                         |
| `cublasCsyrk`           | ✅          | ❌      | ❓      |                         |
| `cublasZsyrk`           | ✅          | ❌      | ❓      |                         |
| `cublasCherk`           | ✅          | ❌      | ❓      |                         |
| `cublasZherk`           | ✅          | ❌      | ❓      |                         |
| `cublasSsyr2k`          | ✅          | ❌      | ❓      |                         |
| `cublasDsyr2k`          | ✅          | ❌      | ❓      |                         |
| `cublasCsyr2k`          | ✅          | ❌      | ❓      |                         |
| `cublasZsyr2k`          | ✅          | ❌      | ❓      |                         |
| `cublasCher2k`          | ✅          | ❌      | ❓      |                         |
| `cublasZher2k`          | ✅          | ❌      | ❓      |                         |
| `cublasSsymm`           | ✅          | ❌      | ❓      |                         |
| `cublasDsymm`           | ✅          | ❌      | ❓      |                         |
| `cublasCsymm`           | ✅          | ❌      | ❓      |                         |
| `cublasZsymm`           | ✅          | ❌      | ❓      |                         |
| `cublasChemm`           | ✅          | ❌      | ❓      |                         |
| `cublasZhemm`           | ✅          | ❌      | ❓      |                         |
| `cublasStrsm`           | ✅          | ❌      | ❓      |                         |
| `cublasDtrsm`           | ✅          | ❌      | ❓      |                         |
| `cublasCtrsm`           | ✅          | ❌      | ❓      |                         |
| `cublasZtrsm`           | ✅          | ❌      | ❓      |                         |
| `cublasStrmm`           | ✅          | ❌      | ❓      |                         |
| `cublasDtrmm`           | ✅          | ❌      | ❓      |                         |
| `cublasCtrmm`           | ✅          | ❌      | ❓      |                         |
| `cublasZtrmm`           | ✅          | ❌      | ❓      |                         |

cuBLAS handle typedefs changed in CUDA 12.


# cuRAND (-lcurand)

| Function                                  | Implemented | Tested  | Working |          Notes           |
| ----------------------------------------- | ----------- | ------- | ------- | -------------------------|
| `curandCreateGenerator`                   | ✅          | ✅      | ✅      |                          |
| `curandCreateGeneratorHost`               | ✅          | ✅      | ✅      |                          |
| `curandSetPseudoRandomGeneratorSeed`      | ✅          | ✅      | ✅      |                          |
| `curandSetQuasiRandomGeneratorDimensions` | ✅          | ✅      | ✅      |            *             |
| `curandGenerate`                          | ✅          | ✅      | ✅      |            *             |
| `curandGenerateUniform`                   | ✅          | ✅      | ✅      |            *             |
| `curandGenerateNormal`                    | ✅          | ✅      | ✅      |            *             |
| `curandGenerateLogNormal`                 | ✅          | ✅      | ✅      |            *             |
| `curandGeneratePoisson`                   | ✅          | ✅      | ✅      |            *             |
| `curandGenerateUniformDouble`             | ✅          | ✅      | ✅      |            *             |
| `curandGenerateNormalDouble`              | ✅          | ✅      | ✅      |            *             |
| `curandGenerateLogNormalDouble`           | ✅          | ✅      | ✅      |            *             |
| `curandGenerateLongLong`                  | ✅          | ✅      | ✅      |            *             |
| `curandGeneratePoisson`                   | ✅          | ✅      | ✅      |            *             |
| `curandDestroyGenerator`                  | ✅          | ✅      | ✅      |                          |

*This function can generate numbers using either a CPU or a GPU generator, created using `curandCreateGenerator` or `curandCreateGeneratorHost`, respectively. **Both CPU and GPU generations are tested and working**.


# cuFFT (-lcufft)

| Function                                  | Implemented | Tested  | Working |          Notes           |
| ----------------------------------------- | ----------- | ------- | ------- | -------------------------|
| `cufftCreate`                             | ✅          | ✅      | ✅      |                          |
| `cufftDestroy`                            | ✅          | ✅      | ✅      |                          |
| `cufftPlan1D`                             | ✅          | ✅      | ✅      |                          |
| `cufftPlan2D`                             | ✅          | ✅      | ✅      |                          |
| `cufftPlan3D`                             | ✅          | ✅      | ✅      |                          |
| `cufftEstimate1D`                         | ✅          | ✅      | ✅      |                          |
| `cufftEstimate2D`                         | ✅          | ✅      | ✅      |                          |
| `cufftEstimate3D`                         | ✅          | ✅      | ✅      |                          |
| `cuFFTEstimateMany`                       | ✅          | ✅      | ✅      |                          |
| `cufftExecC2C`                            | ✅          | ✅      | ✅      |                          |
| `ExecR2C`                                 | ✅          | ✅      | ✅      |                          |
| `ExecC2R`                                 | ✅          | ✅      | ✅      |                          |
| `ExecZ2Z`                                 | ✅          | ✅      | ✅      |                          |
| `ExecD2Z`                                 | ✅          | ✅      | ✅      |                          |
| `ExecZ2D`                                 | ✅          | ✅      | ✅      |                          |
| `cufftMakePlan1D`                         | ✅          | ✅      | ✅      |                          |
| `cufftMakePlan2D`                         | ✅          | ✅      | ✅      |                          |
| `cufftMakePlan3D`                         | ✅          | ✅      | ✅      |                          |
| `cufftMakePlanMany`                       | ✅          | ✅      | ✅      |                          |
| `cufftMakePlanMany64`                     | ✅          | ✅      | ✅      |                          |
| `cufftGetSize1D`                          | ✅          | ✅      | ✅      |                          |
| `cufftGetSize2D`                          | ✅          | ✅      | ✅      |                          |
| `cufftGetSize3D`                          | ✅          | ✅      | ✅      |                          |
| `cufftGetSizeMany`                        | ✅          | ✅      | ✅      |                          |
| `cufftGetSizeMany64`                      | ✅          | ✅      | ✅      |                          |
| `cufftGetSize`                            | ✅          | ✅      | ✅      |                          |
| `cufftSetWorkArea`                        | ✅          | ✅      | ✅      |                          |
| `cufftSetAutoAllocation`                  | ✅          | ✅      | ✅      |                          |
| `cufftGetVersion`                         | ✅          | ✅      | ✅      |                          |
| `cufftSetStream`                          | ✅          | ✅      | ✅      |                          |
| `cufftXtMalloc`                           | ✅          | ✅      | ✅      |                          |
| `cufftXtFree`                             | ✅          | ✅      | ✅      |                          |            
| `cufftXtMemcpy`                           | ✅          | ✅      | ❌      |                          |  
| `cufftXtSetGpus`                          | ✅          | ✅      | ✅      |                          |
| `cufftXtExecDescriptorC2C`                | ✅          | ✅      | ❌      |                          |
| `cufftXtMakePlanMany`                     | ✅          | ✅      | ❌      | Not Supported by GVirtuS |

cuFFT handle typedefs changed in CUDA 12.

# cuDNN (-lcudnn)

| Function                                                   | Implemented | Tested  | Working |          Notes           |
| ---------------------------------------------------------- | ----------- | ------- | ------- | -------------------------|
| `cuDNNCreate`                                              | ✅          | ✅      | ✅      |                          | 
| `cuDNNDestroy`                                             | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetVersion`                                          | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetErrorString`                                      | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetStream`                                           | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetStream`                                           | ✅          | ✅      | ✅      |                          |  
| `cuDNNCreateTensorDescriptor`                              | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetTensor4dDescriptor`                               | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetTensor4dDescriptorEx`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetTensor4dDescriptor`                               | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetTensorNdDescriptor`                               | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetTensorNdDescriptorEx`                             | ✅          | ❌      | ❓      |                          |
| `cuDNNGetTensorNdDescriptor`                               | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetTensorSizeInBytes`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyTensorDescriptor`                             | ✅          | ✅      | ✅      |                          | 
| `cuDNNInitTransformDest`                                   | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateTensorTransformDescriptor`                     | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetTensorTransformDescriptor`                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetTensorTransformDescriptor`                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyTensorTransformDescriptor`                    | ✅          | ❌      | ❓      |                          | 
| `cuDNNTransformTensor`                                     | ✅          | ✅      | ❌      |                          | 
| `cuDNNTransformTensorEx`                                   | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetFoldedConvBackwardDataDescriptors`                | ✅          | ❌      | ❓      |                          | 
| `cuDNNAddTensor`                                           | ✅          | ✅      | ✅      |                          | 
| `cuDNNCreateOpTensorDescriptor`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetOpTensorDescriptor`                               | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetOpTensorDescriptor`                               | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyOpTensorDescriptor`                           | ✅          | ❌      | ❓      |                          | 
| `cuDNNOpTensor`                                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateReduceTensorDescriptor`                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetReduceTensorDescriptor`                           | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetReduceTensorDescriptor`                           | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyReduceTensorDescriptor`                       | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetReductionIndicesSize`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetReductionWorkspaceSize`                           | ✅          | ❌      | ❓      |                          | 
| `cuDNNReduceTensor`                                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetTensor`                                           | ✅          | ❌      | ❓      |                          | 
| `cuDNNScaleTensor`                                         | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateFilterDescriptor`                              | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetFilter4dDescriptor`                               | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetFilter4dDescriptor`                               | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetFilter4dDescriptor_v3`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetFilter4dDescriptor_v3`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetFilter4dDescriptor_v4`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetFilter4dDescriptor_v4`                            | ✅          | ❌      | ❓      |                          |
| `cuDNNSetFilterNdDescriptor`                               | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetFilterNdDescriptor`                               | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetFilterNdDescriptor_v3`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetFilterNdDescriptor_v3`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetFilterNdDescriptor_v4`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetFilterNdDescriptor_v4`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetFilterSizeInBytes`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyFilterDescriptor`                             | ✅          | ✅      | ✅      |                          | 
| `cuDNNTransformFilter`                                     | ✅          | ❌      | ❓      |                          | 
| `cuDNNReorderFilterAndBias`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateConvolutionDescriptor`                         | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetConvolutionMathType`                              | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolutionMathType`                              | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetConvolutionGroupCount`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolutionGroupCount`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetConvolutionReorderType`                           | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolutionReorderType`                           | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetConvolution2dDescriptor`                          | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetConvolution2dDescriptor`                          | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolution2dForwardOutputDim`                    | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetConvolutionNdDescriptor`                          | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolutionNdDescriptor`                          | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolutionNdForwardOutputDim`                    | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyConvolutionDescriptor`                        | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetConvolutionForwardAlgorithmMaxCount`              | ✅          | ❌      | ❓      |                          | 
| `cuDNNFindConvolutionForwardAlgorithm`                     | ✅          | ✅      | ✅      |                          | 
| `cuDNNFindConvolutionForwardAlgorithmEx`                   | ✅          | ✅      | ❌      |                          | 
| `cuDNNGetConvolutionForwardAlgorithm`                      | ✅          | ❌      | ❓      | Deprecated in v8, Use `cuDNNGetConvolutionForwardAlgorithm_v7` instead | 
| `cuDNNGetConvolutionForwardAlgorithm_v7`                   | ✅          | ✅      | ❌      |                          | 
| `cuDNNGetConvolutionForwardWorkspaceSize`                  | ✅          | ✅      | ✅      |                          | 
| `cuDNNConvolutionForward`                                  | ✅          | ✅      | ✅      |                          | 
| `cuDNNConvolutionBiasActivationForward`                    | ✅          | ❌      | ❓      |                          | 
| `cuDNNConvolutionBackwardBias`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolutionBackwardFilterAlgorithmMaxCount`       | ✅          | ❌      | ❓      |                          | 
| `cuDNNFindConvolutionBackwardFilterAlgorithm`              | ✅          | ❌      | ❓      |                          | 
| `cuDNNFindConvolutionBackwardFilterAlgorithmEx`            | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolutionBackwardFilterAlgorithm`               | ✅          | ❌      | ❓      | Deprecated in v8, Use `cuDNNGetConvolutionBackwardFilterAlgorithm_v7` instead | 
| `cuDNNGetConvolutionBackwardFilterAlgorithm_v7`            | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetConvolutionBackwardFilterWorkspaceSize`           | ✅          | ✅      | ✅      |                          | 
| `cuDNNConvolutionBackwardFilter`                           | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetConvolutionBackwardDataAlgorithmMaxCount`         | ✅          | ❌      | ❓      |                          | 
| `cuDNNFindConvolutionBackwardDataAlgorithm`                | ✅          | ❌      | ❓      |                          | 
| `cuDNNFindConvolutionBackwardDataAlgorithmEx`              | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolutionBackwardDataAlgorithm`                 | ✅          | ❌      | ❓      | Deprecated in v8, Use `cuDNNGetConvolutionBackwardDataAlgorithm_v7` instead | 
| `cuDNNGetConvolutionBackwardDataAlgorithm_v7`              | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetConvolutionBackwardDataWorkspaceSize`             | ✅          | ✅      | ✅      |                          | 
| `cuDNNConvolutionBackwardData`                             | ✅          | ✅      | ✅      |                          | 
| `cuDNNIm2Col`                                              | ✅          | ❌      | ❓      |                          | 
| `cuDNNSoftmaxForward`                                      | ✅          | ❌      | ❓      |                          | 
| `cuDNNSoftmaxBackward`                                     | ✅          | ✅      | ✅      |                          | 
| `cuDNNCreatePoolingDescriptor`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetPooling2dDescriptor`                              | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetPooling2dDescriptor`                              | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetPoolingNdDescriptor`                              | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetPoolingNdDescriptor`                              | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetPoolingNdForwardOutputDim`                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetPooling2dForwardOutputDim`                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyPoolingDescriptor`                            | ✅          | ✅      | ✅      |                          | 
| `cuDNNPoolingForward`                                      | ✅          | ✅      | ✅      |                          | 
| `cuDNNPoolingBackward`                                     | ✅          | ✅      | ✅      |                          | 
| `cuDNNCreateActivationDescriptor`                          | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetActivationDescriptor`                             | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetActivationDescriptor`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyActivationDescriptor`                         | ✅          | ✅      | ✅      |                          | 
| `cuDNNActivationForward`                                   | ✅          | ✅      | ✅      |                          | 
| `cuDNNActivationBackward`                                  | ✅          | ✅      | ✅      |                          | 
| `cuDNNCreateLRNDescriptor`                                 | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetLRNDescriptor`                                    | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetLRNDescriptor`                                    | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyLRNDescriptor`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNLRNCrossChannelForward`                              | ✅          | ❌      | ❓      |                          | 
| `cuDNNLRNCrossChannelBackward`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNDivisiveNormalizationForward`                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNDivisiveNormalizationBackward`                       | ✅          | ❌      | ❓      |                          | 
| `cuDNNDeriveBNTensorDescriptor`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetBatchNormalizationForwardTrainingExWorkspaceSize` | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetBatchNormalizationBackwardExWorkspaceSize`        | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetBatchNormalizationTrainingExReserveSpaceSize`     | ✅          | ❌      | ❓      |                          | 
| `cuDNNBatchNormalizationForwardTraining`                   | ✅          | ✅      | ❌      |                          | 
| `cuDNNBatchNormalizationForwardTrainingEx`                 | ✅          | ❌      | ❓      |                          | 
| `cuDNNBatchNormalizationForwardInference`                  | ✅          | ❌      | ❓      |                          | 
| `cuDNNBatchNormalizationBackward`                          | ✅          | ✅      | ❓      |                          | 
| `cuDNNBatchNormalizationBackwardEx`                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateSpatialTransformerDescriptor`                  | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetSpatialTransformerNdDescriptor`                   | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroySpatialTransformerDescriptor`                 | ✅          | ❌      | ❓      |                          | 
| `cuDNNSpatialTfGridGeneratorForward`                       | ✅          | ❌      | ❓      |                          | 
| `cuDNNSpatialTfGridGeneratorBackward`                      | ✅          | ❌      | ❓      |                          | 
| `cuDNNSpatialTfSamplerForward`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNSpatialTfSamplerBackward`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateDropoutDescriptor`                             | ✅          | ✅      | ✅      |                          | 
| `cuDNNDestroyDropoutDescriptor`                            | ✅          | ✅      | ✅      |                          | 
| `cuDNNDropoutGetStatesSize`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNDropoutGetReserveSpaceSize`                          | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetDropoutDescriptor`                                | ✅          | ✅      | ❌      |                          | 
| `cuDNNRestoreDropoutDescriptor`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetDropoutDescriptor`                                | ✅          | ✅      | ❌      |                          | 
| `cuDNNDropoutForward`                                      | ✅          | ❌      | ❓      |                          | 
| `cuDNNDropoutBackward`                                     | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateRNNDescriptor`                                 | ✅          | ✅      | ✅      |                          | 
| `cuDNNDestroyRNNDescriptor`                                | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetRNNDescriptor_v5`                                 | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetRNNDescriptor_v5`                                 | ❌          | ❌      | ❌      |                          | 
| `cuDNNSetRNNDescriptor_v6`                                 | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNRNNBackwardData_v8` instead | 
| `cuDNNGetRNNDescriptor_v6`                                 | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNGetRNNDescriptor_v8` instead | 
| `cuDNNSetRNNDescriptor_v8`                                 | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetRNNDescriptor_v8`                                 | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetRNNMatrixMathType`                                | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNSetRNNDescriptor_v8` instead | 
| `cuDNNGetRNNMatrixMathType`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetRNNBiasMode`                                      | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNSetRNNDescriptor_v8` instead | 
| `cuDNNGetRNNBiasMode`                                      | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNGetRNNDescriptor_v8` instead | 
| `cuDNNRNNSetClip`                                          | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNRNNSetClip_v9` instead | 
| `cuDNNRNNGetClip`                                          | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNRNNGetClip_v9` instead | 
| `cuDNNSetRNNProjectionLayers`                              | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNSetRNNDescriptor_v8` instead | 
| `cuDNNGetRNNProjectionLayers`                              | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNGetRNNDescriptor_v8` instead | 
| `cuDNNCreatePersistentRNNPlan`                             | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cudnnBuildRNNDynamic` instead | 
| `cuDNNDestroyPersistentRNNPlan`                            | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cudnnBuildRNNDynamic` instead | 
| `cuDNNSetPersistentRNNPlan`                                | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cudnnBuildRNNDynamic` instead | 
| `cuDNNGetRNNWorkspaceSize`                                 | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use  `cudnnGetRNNTempSpaceSize` instead | 
| `cuDNNGetRNNTrainingReserveSize`                           | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use  `cudnnGetRNNTempSpaceSize` instead | 
| `cuDNNGetRNNParamsSize`                                    | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use  `cudnnGetRNNWeightSpaceSize` instead | 
| `cuDNNGetRNNLinLayerMatrixParams`                          | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cudnnGetRNNWeightParams` instead | 
| `cuDNNGetRNNLinLayerBiasParams`                            | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cudnnGetRNNWeightParams` instead | 
| `cuDNNRNNForwardInference`                                 | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cudnnRNNForward` instead | 
| `cuDNNRNNForwardTraining`                                  | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cudnnRNNForward` instead | 
| `cuDNNRNNBackwardData`                                     | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNRNNBackwardData_v8` instead | 
| `cuDNNRNNBackwardData_v8`                                  | ❌          | ❌      | ❌      |                          |
| `cuDNNRNNBackwardWeights`                                  | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNSetRNNPaddingMode`                                   | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNRNNBackwardData_v8` instead | 
| `cuDNNGetRNNPaddingMode`                                   | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNGetRNNDescriptor_v8` instead | 
| `cuDNNCreateRNNDataDescriptor`                             | ✅          | ✅      | ✅      |                          | 
| `cuDNNDestroyRNNDataDescriptor`                            | ✅          | ✅      | ✅      |                          | 
| `cuDNNSetRNNDataDescriptor`                                | ✅          | ✅      | ✅      |                          | 
| `cuDNNGetRNNDataDescriptor`                                | ✅          | ✅      | ✅      |                          | 
| `cuDNNRNNForwardTrainingEx`                                | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cudnnRNNForward` instead | 
| `cuDNNRNNForwardInferenceEx`                               | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cudnnRNNForward` instead | 
| `cuDNNRNNBackwardDataEx`                                   | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9, Use `cuDNNRNNBackwardData_v8` instead | 
| `cuDNNRNNBackwardWeightsEx`                                | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNSetRNNAlgorithmDescriptor`                           | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNGetRNNForwardInferenceAlgorithmMaxCount`             | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNFindRNNForwardInferenceAlgorithmEx`                  | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNGetRNNForwardTrainingAlgorithmMaxCount`              | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNFindRNNForwardTrainingAlgorithmEx`                   | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNGetRNNBackwardDataAlgorithmMaxCount`                 | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNFindRNNBackwardDataAlgorithmEx`                      | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNGetRNNBackwardWeightsAlgorithmMaxCount`              | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNFindRNNBackwardWeightsAlgorithmEx`                   | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNCreateSeqDataDescriptor`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroySeqDataDescriptor`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetSeqDataDescriptor`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetSeqDataDescriptor`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateAttnDescriptor`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyAttnDescriptor`                               | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetAttnDescriptor`                                   | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetAttnDescriptor`                                   | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetMultiHeadAttnBuffers`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetMultiHeadAttnWeights`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNMultiHeadAttnForward`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNMultiHeadAttnBackwardData`                           | ✅          | ❌      | ❓      |                          | 
| `cuDNNMultiHeadAttnBackwardWeights`                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateCTCLossDescriptor`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetCTCLossDescriptor`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetCTCLossDescriptorEx`                              | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetCTCLossDescriptor`                                | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetCTCLossDescriptorEx`                              | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyCTCLossDescriptor`                            | ✅          | ❌      | ❓      |                          | 
| `cuDNNCTCLoss`                                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetCTCLossWorkspaceSize`                             | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateAlgorithmDescriptor`                           | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNSetAlgorithmDescriptor`                              | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNGetAlgorithmDescriptor`                              | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNCopyAlgorithmDescriptor`                             | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNDestroyAlgorithmDescriptor`                          | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNCreateAlgorithmPerformance`                          | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNSetAlgorithmPerformance`                             | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNGetAlgorithmPerformance`                             | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNDestroyAlgorithmPerformance`                         | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNGetAlgorithmSpaceSize`                               | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNSaveAlgorithm`                                       | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNRestoreAlgorithm`                                    | ✅          | ❌      | ❓      | Deprecated in v8, Removed in v9 | 
| `cuDNNSetCallback`                                         | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetCallback`                                         | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateFusedOpsConstParamPack`                        | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyFusedOpsConstParamPack`                       | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetFusedOpsConstParamPackAttribute`                  | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetFusedOpsConstParamPackAttribute`                  | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateFusedOpsVariantParamPack`                      | ✅          | ❌      | ❓      |                          | 
| `cuDNNDestroyFusedOpsVariantParamPack`                     | ✅          | ❌      | ❓      |                          | 
| `cuDNNSetFusedOpsVariantParamPackAttribute`                | ✅          | ❌      | ❓      |                          | 
| `cuDNNGetFusedOpsVariantParamPackAttribute`                | ✅          | ❌      | ❓      |                          | 
| `cuDNNCreateFusedOpsPlan`                                  | ✅          | ✅      | ✅      |                          | 
| `cuDNNDestroyFusedOpsPlan`                                 | ✅          | ✅      | ✅      |                          | 
| `cuDNNMakeFusedOpsPlan`                                    | ✅          | ❌      | ❓      |                          | 
| `cuDNNFusedOpsExecute`                                     | ✅          | ❌      | ❓      |                          | 


# cuSPARSE (-lcusparse)

| Function                                  | Implemented | Tested  | Working |          Notes           |
| ----------------------------------------- | ----------- | ------- | ------- | -------------------------|
| `cusparseCreate`                          | ✅          | ❌      | ❓      |                          | 
| `cusparseDestroy`                         | ✅          | ❌      | ❓      |                          | 
| `cusparseGetVersion`                      | ✅          | ❌      | ❓      |                          | 
| `cusparseGetErrorString`                  | ✅          | ❌      | ❓      |                          | 
| `cusparseSetStream`                       | ✅          | ❌      | ❓      |                          | 
| `cusparseGetStream`                       | ✅          | ❌      | ❓      |                          | 
| `cusparseXcsrgemm`                        | ❌          | ❌      | ❌      |                          | 
| `cusparseXcsr2dense`                      | ❌          | ❌      | ❌      |                          | 
| `cusparseXdense2csr`                      | ❌          | ❌      | ❌      |                          | 
| `cusparseXcsrmv`                          | ❌          | ❌      | ❌      |                          | 
| `cusparseXcsrmv_analysis`                 | ❌          | ❌      | ❌      |                          | 
| `cusparseXcsrmv_solve`                    | ❌          | ❌      | ❌      |                          | 


# cuSOLVER (-lcusolver)

| Function                                  | Implemented | Tested  | Working |          Notes           |
| ----------------------------------------- | ----------- | ------- | ------- | -------------------------|
| `cusolverDnCreate`                        | ✅          | ✅      | ✅       |                          |       
| `cusolverDnDestroy`                       | ✅          | ✅      | ✅       |                          | 
| `cusolverDnSetStream`                     | ✅          | ✅      | ✅       |                          | 
| `cusolverDnGetStream`                     | ✅          | ✅      | ✅       |                          |
| `cusolverDnSgetrf`                        | ❌          | ❌      | ❌       |                          |
| `cusolverDnDgetrf`                        | ❌          | ❌      | ❌       |                          |
| `cusolverDnSgetrs`                        | ❌          | ❌      | ❌       |                          |     
| `cusolverDnDgetrs`                        | ❌          | ❌      | ❌       |                          |
| `cusolverDnSgesvd`                        | ❌          | ❌      | ❌       |                          |
| `cusolverDnDgesvd`                        | ❌          | ❌      | ❌       |                          |
| `cusolverDnSpotrf`                        | ❌          | ❌      | ❌       |                          |
| `cusolverDnDpotrf`                        | ❌          | ❌      | ❌       |                          |
