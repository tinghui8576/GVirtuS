#pragma once
#include <cstdint>

namespace gvirtus::communicators {

// 魔数 'G''V''U''X'
static constexpr uint32_t kMagic        = 0x47565558;
static constexpr uint16_t kProtoVersion = 1;

// flags
enum : uint8_t {
    FLAG_EXPECT_RESPONSE = 1 << 0
};

#pragma pack(push,1)
struct ReqHdr {
    uint32_t magic;
    uint16_t version;
    uint8_t  flags;
    uint8_t  reserved;
    uint64_t msg_id;
    uint32_t routine_len;
    uint32_t payload_len;
};

struct RespHdr {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    int32_t  exit_code;
    double   server_exec_sec;
    uint64_t msg_id;
    uint32_t out_len;
};
#pragma pack(pop)

} // namespace gvirtus::communicators
