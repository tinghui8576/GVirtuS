
/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

#include <cstdio>
#include <elf.h>
#include <CudaRt_internal.h>

#include "CudaRt.h"

#define ELF_MAGIC "\177ELF"
#define ELF_MAGIC_SIZE 4

/*
 Routines not found in the cuda's header files.
 KEEP THEM WITH CARE
 */

extern "C" __host__ void **__cudaRegisterFatBinary(void *fatCubin) {
    std::cout << "CudaRegisterFatBinary" << endl;
    /* Fake host pointer */
    __fatBinC_Wrapper_t *bin = (__fatBinC_Wrapper_t *)fatCubin;
     std::cout << "after typecasting to fatBinC" << endl;
    char *raw_data = (char*)bin->data;
    size_t raw_data_size = strlen(raw_data);
    Elf64_Ehdr *eh = nullptr;
    for (size_t offset = 0; offset < 4096; offset++) {
        if (!memcmp(raw_data + offset, ELF_MAGIC, ELF_MAGIC_SIZE)) {
            eh = reinterpret_cast<Elf64_Ehdr *>(raw_data + offset);
            std::cout << "Found ELF header at offset: " << offset << std::endl;
            break;
        }
    }
    if (!eh) {
        cerr << "Error: ELF header not found in the binary." << endl;
        return NULL;
    }

    cout << "Elf header magic: " << eh->e_ident << endl;
    cout << "Found a valid ELF file" << endl;
    /* Section header table :  */
    Elf64_Shdr *sh_table = static_cast<Elf64_Shdr *>(malloc(eh->e_shentsize * eh->e_shnum));
    if (!sh_table) {
        cerr << "Error: Unable to allocate memory for section header table." << endl;
        return NULL;
    }
    byte *baseAddr = (byte *) eh;
    cout << "Base address of ELF file: " << (void *)baseAddr << endl;
    cout << "Loading section header table..." << endl;
    for (uint32_t i = 0; i < eh->e_shnum; i++) {
        Elf64_Shdr tmp;
        size_t offset = eh->e_shoff + i * eh->e_shentsize;
        if (offset + sizeof(Elf64_Shdr) > raw_data_size) { // TODO: Check bounds, find size
            std::cerr << "Section header offset out of bounds!" << std::endl;
            break;
        }
        memcpy(&tmp, baseAddr + offset, sizeof(Elf64_Shdr));
        sh_table[i] = tmp;
    }

    char *sh_str = static_cast<char *>(malloc(sh_table[eh->e_shstrndx].sh_size));
    if (sh_str) {
        memcpy(sh_str, baseAddr + sh_table[eh->e_shstrndx].sh_offset, sh_table[eh->e_shstrndx].sh_size);
        cout << "Section header string table loaded" << endl;
        for (uint32_t i = 0; i < eh->e_shnum; i++) {

            char *szSectionName = (sh_str + sh_table[i].sh_name);
            cout << "Section " << i << ": " << szSectionName << endl;
            if (strncmp(".nv.info.", szSectionName, strlen(".nv.info.")) == 0) {
                char *szFuncName = szSectionName + strlen(".nv.info.");

                byte *p = (byte *) eh + sh_table[i].sh_offset;

                NvInfoFunction infoFunction;
                size_t size;
                NvInfoAttribute *pAttr = (NvInfoAttribute *) p;
                while (pAttr < (NvInfoAttribute *) ((byte *) p + sh_table[i].sh_size)) {
                    size = 0;
                    switch (pAttr->fmt) {
                        case EIFMT_SVAL:
                            cout << "Attribute: " << pAttr->attr << " (SVAL)" << endl;
                            size = sizeof(NvInfoAttribute) + pAttr->value;
                            break;
                        case EIFMT_NVAL:
                            cout << "Attribute: " << pAttr->attr << " (NVAL)" << endl;
                            size = sizeof(NvInfoAttribute);
                            break;
                        case EIFMT_HVAL:
                            cout << "Attribute: " << pAttr->attr << " (HVAL)" << endl;
                            size = sizeof(NvInfoAttribute);
                            break;

                    }
                    if (pAttr->attr == EIATTR_KPARAM_INFO) {
                        NvInfoKParam *nvInfoKParam = (NvInfoKParam *) pAttr;
                        cout << "KParam Info: index=" << nvInfoKParam->index
                                << ", ordinal=" << nvInfoKParam->ordinal
                                << ", offset=" << nvInfoKParam->offset
                                << ", a=" << nvInfoKParam->a
                                << ", size=" << nvInfoKParam->size
                                << ", b=" << nvInfoKParam->b << endl;
                        NvInfoKParam nvInfoKParam1;
                        nvInfoKParam1.index = nvInfoKParam->index;
                        nvInfoKParam1.ordinal = nvInfoKParam->ordinal;
                        nvInfoKParam1.offset = nvInfoKParam->offset;
                        nvInfoKParam1.a = nvInfoKParam->a;
                        nvInfoKParam1.size = nvInfoKParam->size;
                        nvInfoKParam1.b = nvInfoKParam->b;
                        infoFunction.params.push_back(nvInfoKParam1);
                    }
                    pAttr = (NvInfoAttribute *) ((byte *) pAttr + size);
                }
                CudaRtFrontend::addDeviceFunc2InfoFunc(szFuncName, infoFunction);
            }
        }
        free(sh_str);
    }
    free(sh_table);

    cout << "Section header table processed" << endl;
    cout << "Registering fat binary: " << bin->data << endl;
    Buffer *input_buffer = new Buffer();
    input_buffer->AddString(CudaUtil::MarshalHostPointer((void **) bin));
    cout << "Marshalling host pointer: " << bin << endl;
    input_buffer = CudaUtil::MarshalFatCudaBinary(bin, input_buffer);
    cout << "Marshalling fat binary: " << bin->data << endl;
    CudaRtFrontend::Prepare();
    cout << "Just before calling cudaRegisterFatBinary" << endl;
    CudaRtFrontend::Execute("cudaRegisterFatBinary", input_buffer);
    cout << "cudaRegisterFatBinary executed" << endl;
    if (CudaRtFrontend::Success()) return (void **) fatCubin;
    std::cerr << "*** Error in __cudaRegisterFatBinary" << std::endl;
    return NULL;
}

extern "C" __host__ void **__cudaRegisterFatBinaryEnd(void *fatCubin) {
    /* Fake host pointer */
  __fatBinC_Wrapper_t *bin = (__fatBinC_Wrapper_t *)fatCubin;
  char *data = (char *)bin->data;

  Buffer *input_buffer = new Buffer();
  input_buffer->AddString(CudaUtil::MarshalHostPointer((void **)bin));
  input_buffer = CudaUtil::MarshalFatCudaBinary(bin, input_buffer);

  CudaRtFrontend::Prepare();
  CudaRtFrontend::Execute("cudaRegisterFatBinaryEnd", input_buffer);
  if (CudaRtFrontend::Success()) return (void **)fatCubin;
  return NULL;
}

extern "C" __host__ void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    CudaRtFrontend::Execute("cudaUnregisterFatBinary");
}

extern "C" __host__ void __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize) {

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));

    CudaRtFrontend::AddVariableForArguments((gvirtus::common::pointer_t)hostFun);
    CudaRtFrontend::AddStringForArguments(deviceFun);
    CudaRtFrontend::AddStringForArguments(deviceName);
    CudaRtFrontend::AddVariableForArguments(thread_limit);
    CudaRtFrontend::AddHostPointerForArguments(tid);
    CudaRtFrontend::AddHostPointerForArguments(bid);
    CudaRtFrontend::AddHostPointerForArguments(bDim);
    CudaRtFrontend::AddHostPointerForArguments(gDim);
    CudaRtFrontend::AddHostPointerForArguments(wSize);

    CudaRtFrontend::Execute("cudaRegisterFunction");

    deviceFun = CudaRtFrontend::GetOutputString();
    tid = CudaRtFrontend::GetOutputHostPointer<uint3>();
    bid = CudaRtFrontend::GetOutputHostPointer<uint3>();
    bDim = CudaRtFrontend::GetOutputHostPointer<dim3>();
    gDim = CudaRtFrontend::GetOutputHostPointer<dim3>();
    wSize = CudaRtFrontend::GetOutputHostPointer<int>();

    CudaRtFrontend::addHost2DeviceFunc((void*)hostFun,deviceFun);
}

extern "C" __host__ void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                           char *deviceAddress,
                                           const char *deviceName, int ext,
                                           int size, int constant, int global) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    CudaRtFrontend::AddStringForArguments(hostVar);
    CudaRtFrontend::AddStringForArguments(deviceAddress);
    CudaRtFrontend::AddStringForArguments(deviceName);
    CudaRtFrontend::AddVariableForArguments(ext);
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddVariableForArguments(constant);
    CudaRtFrontend::AddVariableForArguments(global);
    // cout << "RegisterVar: fatCubinHandle: " << fatCubinHandle
    //      << ", hostVar: " << hostVar
    //      << ", deviceAddress: " << deviceAddress
    //      << ", deviceName: " << deviceName
    //      << ", ext: " << ext
    //      << ", size: " << size
    //      << ", constant: " << constant
    //      << ", global: " << global << endl;
    CudaRtFrontend::Execute("cudaRegisterVar");
}

extern "C" __host__ void __cudaRegisterShared(void **fatCubinHandle,
                                              void **devicePtr) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    CudaRtFrontend::AddStringForArguments((char *)devicePtr);
    CudaRtFrontend::Execute("cudaRegisterShared");
}

extern "C" __host__ void __cudaRegisterSharedVar(void **fatCubinHandle,
                                                 void **devicePtr, size_t size,
                                                 size_t alignment,
                                                 int storage) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    CudaRtFrontend::AddStringForArguments((char *)devicePtr);
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddVariableForArguments(alignment);
    CudaRtFrontend::AddVariableForArguments(storage);
    CudaRtFrontend::Execute("cudaRegisterSharedVar");
}

extern "C" __host__ int __cudaSynchronizeThreads(void **x, void *y) {
  // FIXME: implement
  std::cerr << "*** Error: __cudaSynchronizeThreads() not yet implemented!"
            << std::endl;
  return 0;
}

#if CUDA_VERSION >= 9020
    extern "C" __host__ __device__ unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
        CudaRtFrontend::Prepare();
        CudaRtFrontend::AddVariableForArguments(gridDim);
        CudaRtFrontend::AddVariableForArguments(blockDim);
        CudaRtFrontend::AddVariableForArguments(sharedMem);
        CudaRtFrontend::AddDevicePointerForArguments(stream);

        CudaRtFrontend::Execute("cudaPushCallConfiguration");

        return CudaRtFrontend::GetExitCode();
    }


    extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim,
                                                                dim3 *blockDim,
                                                                size_t *sharedMem,
                                                                cudaStream_t *stream) {
        CudaRtFrontend::Prepare();

        CudaRtFrontend::Execute("cudaPopCallConfiguration");

        *gridDim = CudaRtFrontend::GetOutputVariable<dim3>();
        *blockDim = CudaRtFrontend::GetOutputVariable<dim3>();
        *sharedMem = CudaRtFrontend::GetOutputVariable<size_t>();
        cudaStream_t stream1 = CudaRtFrontend::GetOutputVariable<cudaStream_t>();

        memcpy(stream, &stream1, sizeof(cudaStream_t));
        return CudaRtFrontend::GetExitCode();
    }
#endif
