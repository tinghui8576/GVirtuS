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
#include <unordered_map> // <-- 新增: 用于Waiter管理

#include <ucp/api/ucp.h>
#include <gvirtus/communicators/Communicator.h>

// <-- 新增: 前向声明，用于 friend 声明
namespace gvirtus { namespace backend { class Process; } }

namespace gvirtus {
namespace communicators {

class UcxCommunicator : public Communicator {
public:
    // ... [协议常量、结果类型等保持不变] ...
    static constexpr uint32_t kMagic        = 0x47565558u;
    static constexpr uint16_t kProtoVersion = 1;
    enum : uint8_t { FLAG_EXPECT_RESPONSE = 1u << 0 };
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
    // 构造函数的行为将根据上下文变得非对称：
    // - 客户端: 创建完整的流水线 (发送/接收/进度线程)
    // - 服务器端: 创建一个轻量级的 ep 包装器，不创建自己的线程
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

    // ---- 流水线请求接口 (外部API保持不变) ----
    SubmitResult SubmitRequest(const char* routine,
                               const void* payload, size_t payload_len,
                               bool expect_response = true);

    AsyncTicket SubmitRequestAsync(const char* routine,
                                   const void* payload, size_t payload_len);
    
    // ---- 线程管理 ----
    // 旧的 Start/StopNetwork() 将被更有描述性的 Start/StopPipeline() 取代
    void StartPipeline();
    void StopPipeline();
    // 进度线程管理保持不变
    void StartProgress();
    void StopProgress();

private:
    // <-- 新增: 授予后端Process访问底层收发函数的权限
    // 这样服务器端可以直接、高效地发送响应，而无需通过发送队列。
    friend class gvirtus::backend::Process;

    // ... [UCX 对象, _req_state, Queued 结构体保持不变] ...
    ucp_context_h  _context  = nullptr;
    ucp_worker_h   _worker   = nullptr;
    ucp_listener_h _listener = nullptr;
    ucp_ep_h       _ep       = nullptr;
    struct _req_state;
    struct Queued;

    // ---- 内部状态 (保持不变) ----
    bool _is_server_side_wrapper = false;
    std::atomic<bool> _closing{false};
    char        _hostname[256] = {0};
    char        _port[32]      = {0};
    std::string _tls;
    // ... [listener 相关的成员保持不变] ...
    struct sockaddr_storage _listen_addr{};
    socklen_t               _listen_addrlen = 0;
    std::string _bind_ifname;
    mutable std::mutex              _accept_mtx;
    mutable std::condition_variable _accept_cv;
    mutable std::queue<ucp_ep_h>    _accepted_eps;

    // ---- 进度线程 (保持不变) ----
    std::atomic<bool> _progress_running{false};
    std::thread       _progress_thread;
    std::mutex        _progress_mtx;
    
    // ========================================================================
    //                         核心修改区域
    // ========================================================================
    
    // ---- Waiter 结构体 (保持不变) ----
    struct Waiter;
    
    // ---- 1. 新增: Waiter 管理中心 ("总服务台") ----
    // 用于存储所有正在等待响应的同步请求。
    // key: msg_id, value: 对应的 Waiter 对象
    std::mutex _waiters_mtx;
    std::unordered_map<uint64_t, std::shared_ptr<Waiter>> _waiters;

    // ---- 2. 重构: 流水线线程 (收发分离) ----
    // 将原先的 _net_thread 明确为发送线程，并新增一个接收线程。
    
    // 发送线程: 负责从 _sendq 取出请求并发送
    std::thread                       _send_thread;
    std::mutex                        _send_thread_mtx;
    std::atomic<bool>                 _send_thread_running{false};

    // 接收线程: 负责接收所有响应，并通过 _waiters 唤醒等待的线程
    std::thread                       _recv_thread;
    std::mutex                        _recv_thread_mtx;
    std::atomic<bool>                 _recv_thread_running{false};

    // ---- 发送队列 (保持不变) ----
    std::mutex                        _sendq_mtx;
    std::condition_variable           _sendq_cv;
    std::queue<std::shared_ptr<Queued>> _sendq;
    std::atomic<bool>                 _sendq_stopping{false};

    // ---- 消息ID生成器 (保持不变) ----
    std::atomic<uint64_t>             _msg_id_gen{0};
    uint64_t _next_msg_id();
    
    // ========================================================================

    // ---- 内部工具 (大部分保持不变) ----
    void _init_context();
    void _finalize_context();
    void _create_worker();
    void _destroy_worker();
    void _setup_listener();
    void _destroy_listener();
    void _resolve_sockaddr(struct sockaddr_storage& ss, socklen_t& slen) const;

    // 底层收发函数，现在也将被服务器端直接使用 (通过 friend)
    void   _recv_stream_exact(ucp_worker_h worker, ucp_ep_h ep, void* buf, size_t nbytes);
    void   _send_stream_exact(ucp_ep_h ep, const void* buf, size_t nbytes);

    // ---- 3. 重构: 线程循环函数 ----
    // 原有的 _network_loop 将被重命名并改造为 _send_loop
    void _send_loop();
    // 新增 _recv_loop 用于接收线程
    void _recv_loop();

    // 回调函数 (保持不变)
    static void _on_conn_request(ucp_conn_request_h req, void* arg);
    static void _on_ep_error(void* arg, ucp_ep_h ep, ucs_status_t status);
};

} // namespace communicators
} // namespace gvirtus

// ... [create_communicator 声明保持不变] ...

extern "C"
std::shared_ptr<gvirtus::communicators::UcxCommunicator>
create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> end);
