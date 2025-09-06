// Endpoint_Ucx.cpp
#include "gvirtus/communicators/Endpoint_Ucx.h"
#include "gvirtus/communicators/EndpointFactory.h"
#include <regex>

using namespace gvirtus::communicators;

Endpoint_Ucx::Endpoint_Ucx(const std::string &endp_suite,
                           const std::string &endp_protocol,
                           const std::string &endp_address,
                           const std::string &endp_port) {
    suite(endp_suite);
    protocol(endp_protocol);
    address(endp_address);
    port(endp_port);
}

Endpoint &Endpoint_Ucx::suite(const std::string &s) {
    // 允许 "ucx" 或 "ucx-<alpha>"
    std::regex pattern{R"([[:alpha:]]+(-[[:alpha:]]+)?)"};
    std::smatch m;
    std::regex_search(s, m, pattern);
    if (!m.empty() && s == m[0]) _suite = s;
    return *this;
}

Endpoint &Endpoint_Ucx::protocol(const std::string &p) {
    // 仅字母：如 "ucp" / "sockaddr"
    std::regex pattern{R"([[:alpha:]]+)"};
    std::smatch m;
    std::regex_search(p, m, pattern);
    if (!m.empty() && p == m[0]) _protocol = p;
    return *this;
}

Endpoint_Ucx &Endpoint_Ucx::address(const std::string &addr) {
    // 与 RDMA 相同：IPv4
    std::regex pattern{
        R"(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)"};
    std::smatch m;
    std::regex_search(addr, m, pattern);
    if (!m.empty() && addr == m[0]) _address = addr;
    return *this;
}

Endpoint_Ucx &Endpoint_Ucx::port(const std::string &p) {
    // 与 RDMA 相同的端口校验
    std::regex pattern{
        R"((6553[0-5]|655[0-2][0-9]\d|65[0-4](\d){2}|6[0-4](\d){3}|[1-5](\d){4}|[1-9](\d){0,3}))"};
    std::smatch m;
    std::regex_search(p, m, pattern);
    if (!m.empty() && p == m[0]) _port = static_cast<uint16_t>(std::stoi(p));
    return *this;
}

void gvirtus::communicators::from_json(const nlohmann::json &j, Endpoint_Ucx &end) {
    // 与 RDMA 的 JSON 结构保持一致
    auto el = j["communicator"][EndpointFactory::index()]["endpoint"];
    end.suite(el.at("suite"));
    end.protocol(el.at("protocol"));
    end.address(el.at("server_address"));
    end.port(el.at("port"));
}
