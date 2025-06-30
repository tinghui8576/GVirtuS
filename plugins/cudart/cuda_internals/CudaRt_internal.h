//
// Created by Raffaele Montella on 06/11/21.
// Edited by Theodoros Aslanidis on 22/06/25.
//

#ifndef GVIRTUS_CUDART_INTERNAL_H
#define GVIRTUS_CUDART_INTERNAL_H

#include <elf.h>
#include <driver_types.h>
#include <vector>
#include <string>

#define EIFMT_NVAL 0x01
#define EIFMT_HVAL 0x03
#define EIFMT_SVAL 0x04

#define EIATTR_ERROR 0x00
#define EIATTR_PAD 0x01
#define EIATTR_IMAGE_SLOT 0x02
#define EIATTR_JUMPTABLE_RELOCS 0x03
#define EIATTR_CTAIDZ_USED 0x04
#define EIATTR_MAX_THREADS 0x05
#define EIATTR_IMAGE_OFFSET 0x06
#define EIATTR_IMAGE_SIZE 0x07
#define EIATTR_TEXTURE_NORMALIZED 0x08
#define EIATTR_SAMPLER_INIT 0x09
#define EIATTR_PARAM_CBANK 0x0a
#define EIATTR_SMEM_PARAM_OFFSETS 0x0b
#define EIATTR_CBANK_PARAM_OFFSETS 0x0c
#define EIATTR_SYNC_STACK 0x0d
#define EIATTR_TEXID_SAMPID_MAP 0x0e
#define EIATTR_EXTERNS 0x0f
#define EIATTR_REQNTID 0x10
#define EIATTR_FRAME_SIZE 0x11
#define EIATTR_MIN_STACK_SIZE 0x12
#define EIATTR_SAMPLER_FORCE_UNNORMALIZED 0x13
#define EIATTR_BINDLESS_IMAGE_OFFSETS 0x14
#define EIATTR_BINDLESS_TEXTURE_BANK 0x15
#define EIATTR_BINDLESS_SURFACE_BANK 0x16
#define EIATTR_KPARAM_INFO 0x17
#define EIATTR_SMEM_PARAM_SIZE 0x18
#define EIATTR_CBANK_PARAM_SIZE 0x19
#define EIATTR_QUERY_NUMATTRIB 0x1a
#define EIATTR_MAXREG_COUNT 0x1b
#define EIATTR_EXIT_INSTR_OFFSETS 0x1c
#define EIATTR_S2RCTAID_INSTR_OFFSETS 0x1d
#define EIATTR_CRS_STACK_SIZE 0x1e
#define EIATTR_NEED_CNP_WRAPPER 0x1f
#define EIATTR_NEED_CNP_PATCH 0x20
#define EIATTR_EXPLICIT_CACHING 0x21
#define EIATTR_ISTYPEP_USED 0x22
#define EIATTR_MAX_STACK_SIZE 0x23
#define EIATTR_SUQ_USED 0x24
#define EIATTR_LD_CACHEMOD_INSTR_OFFSETS 0x25
#define EIATTR_LOAD_CACHE_REQUEST 0x26
#define EIATTR_ATOM_SYS_INSTR_OFFSETS 0x27
#define EIATTR_COOP_GROUP_INSTR_OFFSETS 0x28
#define EIATTR_COOP_GROUP_MASK_REGIDS 0x29
#define EIATTR_SW1850030_WAR 0x2a
#define EIATTR_WMMA_USED 0x2b
#define EIATTR_SW2393858_WAR 0x30
#define EIATTR_CUDA_API_VERSION 0x37


/////// This is for a newer version of the NVInfo format /////////////////////////////////////////////////////////////////////////////////
// #pragma pack(push, 1)

// enum NVInfoFormat : uint8_t {
//     EIFMT_NVAL = 0x01,
//     EIFMT_BVAL,
//     EIFMT_HVAL,
//     EIFMT_SVAL
// };

// enum NVInfoAttribute : uint8_t {
//     EIATTR_ERROR = 0x00,
//     EIATTR_PAD,
//     EIATTR_IMAGE_SLOT,
//     EIATTR_JUMPTABLE_RELOCS,
//     EIATTR_CTAIDZ_USED,
//     EIATTR_MAX_THREADS,
//     EIATTR_IMAGE_OFFSET,
//     EIATTR_IMAGE_SIZE,
//     EIATTR_TEXTURE_NORMALIZED,
//     EIATTR_SAMPLER_INIT,
//     EIATTR_PARAM_CBANK,
//     EIATTR_SMEM_PARAM_OFFSETS,
//     EIATTR_CBANK_PARAM_OFFSETS,
//     EIATTR_SYNC_STACK,
//     EIATTR_TEXID_SAMPID_MAP,
//     EIATTR_EXTERNS,
//     EIATTR_REQNTID,
//     EIATTR_FRAME_SIZE,
//     EIATTR_MIN_STACK_SIZE,
//     EIATTR_SAMPLER_FORCE_UNNORMALIZED,
//     EIATTR_BINDLESS_IMAGE_OFFSETS,
//     EIATTR_BINDLESS_TEXTURE_BANK,
//     EIATTR_BINDLESS_SURFACE_BANK,
//     EIATTR_KPARAM_INFO,
//     EIATTR_SMEM_PARAM_SIZE,
//     EIATTR_CBANK_PARAM_SIZE,
//     EIATTR_QUERY_NUMATTRIB,
//     EIATTR_MAXREG_COUNT,
//     EIATTR_EXIT_INSTR_OFFSETS,
//     EIATTR_S2RCTAID_INSTR_OFFSETS,
//     EIATTR_CRS_STACK_SIZE,
//     EIATTR_NEED_CNP_WRAPPER,
//     EIATTR_NEED_CNP_PATCH,
//     EIATTR_EXPLICIT_CACHING,
//     EIATTR_ISTYPEP_USED,
//     EIATTR_MAX_STACK_SIZE,
//     EIATTR_SUQ_USED,
//     EIATTR_LD_CACHEMOD_INSTR_OFFSETS,
//     EIATTR_LOAD_CACHE_REQUEST,
//     EIATTR_ATOM_SYS_INSTR_OFFSETS,
//     EIATTR_COOP_GROUP_INSTR_OFFSETS,
//     EIATTR_COOP_GROUP_MAX_REGIDS,
//     EIATTR_SW1850030_WAR,
//     EIATTR_WMMA_USED,
//     EIATTR_HAS_PRE_V10_OBJECT,
//     EIATTR_ATOMF16_EMUL_INSTR_OFFSETS,
//     EIATTR_ATOM16_EMUL_INSTR_REG_MAP,
//     EIATTR_REGCOUNT,
//     EIATTR_SW2393858_WAR,
//     EIATTR_INT_WARP_WIDE_INSTR_OFFSETS,
//     EIATTR_SHARED_SCRATCH,
//     EIATTR_STATISTICS,

//     // New between cuda 10.2 and 11.6
//     EIATTR_INDIRECT_BRANCH_TARGETS,
//     EIATTR_SW2861232_WAR,
//     EIATTR_SW_WAR,
//     EIATTR_CUDA_API_VERSION,
//     EIATTR_NUM_MBARRIERS,
//     EIATTR_MBARRIER_INSTR_OFFSETS,
//     EIATTR_COROUTINE_RESUME_ID_OFFSETS,
//     EIATTR_SAM_REGION_STACK_SIZE,
//     EIATTR_PER_REG_TARGET_PERF_STATS,

//     // New between cuda 11.6 and 11.8
//     EIATTR_CTA_PER_CLUSTER,
//     EIATTR_EXPLICIT_CLUSTER,
//     EIATTR_MAX_CLUSTER_RANK,
//     EIATTR_INSTR_REG_MAP,
// };

// will work with the new parseNvInfoSections function when fixed
// // This is the header of each item in .nv.info.*
// typedef struct {
//     NVInfoFormat format;
//     NVInfoAttribute attribute;
// } NVInfoItemHeader;

// // For SVAL
// typedef struct  {
//     uint16_t value_size;
// } NVInfoSvalHeader;

// typedef struct {
//     uint32_t index;
//     uint16_t ordinal;
//     uint16_t offset;
//     uint32_t tmp;
//     uint8_t log_alignment() const { return tmp & 0xFF; }
//     uint8_t space()         const { return (tmp >> 8) & 0xF; }
//     uint8_t cbank()         const { return (tmp >> 12) & 0x1F; }
//     bool    is_cbank()      const { return ((tmp >> 16) & 2) == 0; }
//     uint16_t size_bytes()   const { return (((tmp >> 16) & 0xFFFF) >> 2); }
// } NVInfoKParamInfoValue;

// typedef struct {
//     NVInfoFormat format;
//     NVInfoAttribute attribute;
//     enum Type : uint16_t { NO_VALUE, BVAL, HVAL, KPARAM_INFO, EXTERN, OTHER } type;
//     uint16_t uval;
//     NVInfoKParamInfoValue kparam;
//     std::string extern_name;
// } NVInfoItem;

// typedef struct {
//     std::vector<NVInfoKParamInfoValue> params;
// } NVInfoFunction;

// #pragma pack(pop)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// original gvirtus ones
typedef struct {
    uint8_t fmt;
    uint8_t attr;
    uint16_t value;
} NvInfoAttribute;

typedef struct {
    uint32_t index;
    uint16_t ordinal;
    uint16_t offset;
    uint32_t tmp;
    uint8_t log_alignment() const { return tmp & 0xFF; }
    uint8_t space()         const { return (tmp >> 8) & 0xF; }
    uint8_t cbank()         const { return (tmp >> 12) & 0x1F; }
    bool    is_cbank()      const { return ((tmp >> 16) & 2) == 0; }
    uint16_t size_bytes()   const { return (((tmp >> 16) & 0xFFFF) >> 2); }
} NvInfoKParam;

// typedef struct {
//     NvInfoAttribute nvInfoAttribute;
//     uint16_t index;
//     uint16_t align;
//     uint16_t ordinal;
//     uint16_t offset;
//     uint16_t a;
//     uint8_t size;
//     uint8_t b;
// } NvInfoKParam;

typedef struct __infoFunction {
    std::vector<NvInfoKParam> params;
} NvInfoFunction;

// typedef struct __infoFunctionEx { // not needed
//     NvInfoFunction infoFunction;
//     cudaStream_t stream;
//     bool adHocStream;
//     void **args;
// } NvInfoFunctionEx;

#endif //GVIRTUS_CUDART_INTERNAL_H
