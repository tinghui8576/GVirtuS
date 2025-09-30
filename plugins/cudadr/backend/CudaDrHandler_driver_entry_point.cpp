/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Ting-Hui Cheng <tinghc@es.aau.dk>,
 *             Department of Electronic Systems, Aalborg University, Denmark
 */

#include "CudaDrHandler.h"

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

CUDA_DRIVER_HANDLER(GetProcAddress) {

    try{
        // char *symbol = input_buffer->AssignString();
        char *symbol = input_buffer->Assign<char>();
        // = pThis->GetSymbol(input_buffer.get());
        void* pfn = nullptr;  
        int cudaVersion = input_buffer->Get<int>();
        cuuint64_t flags = input_buffer->Get<cuuint64_t>();
        CUdriverProcAddressQueryResult symbolStatus;

        CUresult exit_code = cuGetProcAddress(symbol, &pfn, cudaVersion, flags, &symbolStatus);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->AddMarshal(pfn);
        out->Add(symbolStatus);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(CUDA_ERROR_UNKNOWN, std::make_shared<Buffer>());
    }
}
