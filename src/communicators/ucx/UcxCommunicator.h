#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <ucp/api/ucp.h>
#include <gvirtus/communicators/Communicator.h>

namespace gvirtus {
namespace communicators {

class UcxCommunicator : public Communicator {
public:
    // ---- 协议常量 ----
    static constexpr uint32_t kMagic        = 0x47565558u;  // 'GVUX'
    static constexpr uint16_t kProtoVersion = 1;

    enum : uint8_t {
        FLAG_EXPECT_RESPONSE = 1u << 0,
        FLAG_ASYNC           = 1u << 1
    };

    // ---- 协议头 ----


    // ---- 结果类型 ----
    struct SubmitResult {
        ucs_status_t    transport_status{UCS_OK};
        int             exit_code{0};
        double          server_exec_sec{0.0};
        std::vector<char> out;
    };

    struct AsyncTicket {
        uint64_t msg_id{0};
    };

    // ---- 构造/析构 ----
    UcxCommunicator(const std::string& hostname, const std::string& port);
    UcxCommunicator(const std::string& hostname, const std::string& port, const std::string& tls);
    UcxCommunicator(ucp_context_h context, ucp_worker_h worker, ucp_ep_h ep);
    ~UcxCommunicator() override;

    // ---- Communicator 基础接口 ----
    void   Serve() override;
    void   Connect() override;
    size_t Read(char* buffer, size_t size) override;
    size_t Write(const char* buffer, size_t size) override;
    void   Sync() override;
    void   Close() override;
    const Communicator* const Accept() const override;
    std::string to_string() override;

    // ---- 新接口：流水线请求 ----
    SubmitResult SubmitRequest(const char* routine,
                               const void* payload, size_t payload_len,
                               bool expect_response = true);

    AsyncTicket SubmitRequestAsync(const char* routine,
                                   const void* payload, size_t payload_len);

    // ---- 进度/网络线程管理 ----
    void ProgressOnce();
    void StartProgress();
    void StopProgress();

    void StartNetwork();
    void StopNetwork();

private:


    // ---- UCX 对象 ----
    ucp_context_h  _context  = nullptr;
    ucp_worker_h   _worker   = nullptr;
    ucp_listener_h _listener = nullptr;
    ucp_ep_h       _ep       = nullptr;
    
    struct _req_state;   // 前置声明
    struct Waiter;       // 前置声明
    struct Queued;       // 前置声明
    // ---- 状态 ----
    bool _is_server_side_wrapper = false;
    std::atomic<bool> _closing{false};
    char        _hostname[256] = {0};
    char        _port[32]      = {0};
    std::string _tls;

    // listener
    struct sockaddr_storage _listen_addr{};
    socklen_t               _listen_addrlen = 0;
    std::string _bind_ifname;
    mutable std::mutex              _accept_mtx;
    mutable std::condition_variable _accept_cv;
    mutable std::queue<ucp_ep_h>    _accepted_eps;

    // worker
    std::atomic<bool> _progress_running{false};
    std::thread       _progress_thread;
    std::mutex        _progress_mtx;

// ---- 发送队列与网络线程 ----
std::thread                       _net_thread;
std::mutex                        _net_mtx;
std::atomic<bool>                 _net_running{false};

std::mutex                        _sendq_mtx;
std::condition_variable           _sendq_cv;
std::queue<std::shared_ptr<UcxCommunicator::Queued>> _sendq;
std::atomic<bool>                 _sendq_stopping{false};

// 消息ID生成器
std::atomic<uint64_t>             _msg_id_gen{0};
uint64_t _next_msg_id();

    // 内部工具
    void _init_context();
    void _finalize_context();
    void _create_worker();
    void _destroy_worker();
    void _setup_listener();
    void _destroy_listener();
    void _progress_once() const;
    void _resolve_sockaddr(struct sockaddr_storage& ss, socklen_t& slen) const;

    void   _recv_stream_exact(ucp_worker_h worker, ucp_ep_h ep, void* buf, size_t nbytes);
    void   _send_stream_exact(ucp_ep_h ep, const void* buf, size_t nbytes);

    void _network_loop();

    static void _on_conn_request(ucp_conn_request_h req, void* arg);
    static void _on_ep_error(void* arg, ucp_ep_h ep, ucs_status_t status);
};

} // namespace communicators
} // namespace gvirtus

extern "C"
std::shared_ptr<gvirtus::communicators::UcxCommunicator>
create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> end);
