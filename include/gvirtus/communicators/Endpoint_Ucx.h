// Endpoint_Ucx.h
#ifndef GVIRTUS_ENDPOINT_UCX_H
#define GVIRTUS_ENDPOINT_UCX_H

#pragma once

#include <nlohmann/json.hpp>
#include "Endpoint.h"
#include <string>
#include <cstdint>

namespace gvirtus::communicators {

class Endpoint_Ucx : public Endpoint {
private:
    std::string  _suite;     // e.g. "ucx-rc", "ucx-tcp", "ucx"
    std::string  _protocol;  // e.g. "ucp", "sockaddr"
    std::string  _address;   // IPv4 (与 RDMA 一致的校验)
    std::uint16_t _port;

public:
    Endpoint_Ucx() = default;

    explicit Endpoint_Ucx(const std::string &endp_suite,
                          const std::string &endp_protocol,
                          const std::string &endp_address,
                          const std::string &endp_port);

    // 便捷构造（与 RDMA 对齐）
    explicit Endpoint_Ucx(const std::string &endp_suite)
        : Endpoint_Ucx(endp_suite, "ucp", "127.0.0.1", "9999") {}

    // Getters（与 RDMA 一致）
    inline const std::string &suite() const { return _suite; }
    inline const std::string &protocol() const { return _protocol; }
    inline const std::string &address() const { return _address; }
    inline const std::uint16_t &port() const { return _port; }

    // Setters（返回类型与 RDMA 对齐）
    Endpoint &suite(const std::string &suite) override;
    Endpoint &protocol(const std::string &protocol) override;
    Endpoint_Ucx &address(const std::string &address);
    Endpoint_Ucx &port(const std::string &port);

    virtual inline const std::string to_string() const override {
        return _suite + ":" + _protocol + "://" + _address + ":" + std::to_string(_port);
    }

    friend void from_json(const nlohmann::json &j, Endpoint_Ucx &end);
};

} // namespace gvirtus::communicators

#endif // GVIRTUS_ENDPOINT_UCX_H
