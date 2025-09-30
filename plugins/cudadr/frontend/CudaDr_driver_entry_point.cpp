/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Ting-Hui Cheng <tinghc@es.aau.dk>,
 *             Department of Electronic Systems, Aalborg University, Denmark
 */

#include "CudaDr.h"

using namespace std;

/*Returns the compute capability of the device*/
extern "C" CUresult cuGetProcAddress(const char* symbol, void** pfn, int  cudaVersion, 
                                    cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus ) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(symbol);
    // CudaDrFrontend::AddDevicePointerForArguments(pfn);
    CudaDrFrontend::AddVariableForArguments(cudaVersion);
    CudaDrFrontend::AddVariableForArguments(flags);
    if (symbolStatus != NULL) {
        CudaDrFrontend::AddHostPointerForArguments(symbolStatus);
    }
    CudaDrFrontend::Execute("cuGetProcAddress");
    if (CudaDrFrontend::Success()) {
        *pfn = CudaDrFrontend::GetOutputDevicePointer();
        if (symbolStatus != NULL) {
            *symbolStatus = CudaDrFrontend::GetOutputVariable<CUdriverProcAddressQueryResult>();
        }
    }
    return CudaDrFrontend::GetExitCode();
}


