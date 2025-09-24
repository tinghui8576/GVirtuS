#pragma once
#include <cstdint>
#include <arpa/inet.h>

namespace gvirtus::communicators {

// ======================================================================
// [最终修正] 添加对有符号整数 (signed) 的支持
// ======================================================================

// --- Host to Network ---
template<class T> T hton_any(T v);
// Unsigned types
template<> inline uint16_t hton_any<uint16_t>(uint16_t v){ return htons(v); }
template<> inline uint32_t hton_any<uint32_t>(uint32_t v){ return htonl(v); }
template<> inline uint64_t hton_any<uint64_t>(uint64_t v){
  uint64_t hi = htonl(static_cast<uint32_t>(v >> 32));
  uint64_t lo = htonl(static_cast<uint32_t>(v & 0xffffffffu));
  return (lo << 32) | hi;
}
// Signed types (newly added)
template<> inline int16_t hton_any<int16_t>(int16_t v) {
    return static_cast<int16_t>(htons(static_cast<uint16_t>(v)));
}
template<> inline int32_t hton_any<int32_t>(int32_t v) {
    return static_cast<int32_t>(htonl(static_cast<uint32_t>(v)));
}
template<> inline int64_t hton_any<int64_t>(int64_t v) {
    return static_cast<int64_t>(hton_any<uint64_t>(static_cast<uint64_t>(v)));
}


// --- Network to Host ---
template<class T> T ntoh_any(T v);
// Unsigned types
template<> inline uint16_t ntoh_any<uint16_t>(uint16_t v){ return ntohs(v); }
template<> inline uint32_t ntoh_any<uint32_t>(uint32_t v){ return ntohl(v); }
template<> inline uint64_t ntoh_any<uint64_t>(uint64_t v){
  uint64_t hi = ntohl(static_cast<uint32_t>(v >> 32));
  uint64_t lo = ntohl(static_cast<uint32_t>(v & 0xffffffffu));
  return (lo << 32) | hi;
}
// Signed types (newly added)
template<> inline int16_t ntoh_any<int16_t>(int16_t v) {
    return static_cast<int16_t>(ntohs(static_cast<uint16_t>(v)));
}
template<> inline int32_t ntoh_any<int32_t>(int32_t v) {
    return static_cast<int32_t>(ntohl(static_cast<uint32_t>(v)));
}
template<> inline int64_t ntoh_any<int64_t>(int64_t v) {
    return static_cast<int64_t>(ntoh_any<uint64_t>(static_cast<uint64_t>(v)));
}

// ======================================================================

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