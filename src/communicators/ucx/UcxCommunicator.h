#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <ucp/api/ucp.h>

#include <gvirtus/communicators/Communicator.h>

namespace gvirtus {
namespace communicators {

class UcxCommunicator : public Communicator {
public:
    // 构造：前端使用（客户端）
    UcxCommunicator(const std::string& hostname, const std::string& port);
    UcxCommunicator(const std::string& hostname, const std::string& port, const std::string& tls);

    // 构造：后端 Accept() 返回的服务端包装（不销毁 context/worker）
    UcxCommunicator(ucp_context_h context, ucp_worker_h worker, ucp_ep_h ep);

    ~UcxCommunicator() override;

    // gvirtus 抽象接口
    void   Serve() override;   // 后端监听
    void   Connect() override; // 前端连接
    size_t Read(char* buffer, size_t size) override;
    size_t Write(const char* buffer, size_t size) override;
    void   Sync() override;
    void   Close() override;

    // 后端：阻塞等待新的连接，返回一个新的 Communicator*
    // MODIFIED: Restored 'const' qualifiers to match the base class and user requirement.
    const gvirtus::communicators::Communicator* const Accept() const override;

private:
    // --- UCX 基本对象 ---
    ucp_context_h  _context = nullptr;
    ucp_worker_h   _worker  = nullptr;
    ucp_listener_h _listener = nullptr;
    ucp_ep_h       _ep = nullptr;

    // --- 运行时配置/状态 ---
    bool  _is_server_side_wrapper = false; // Accept() 返回的包装对象不负责销毁 context/worker
    bool  _closing = false;

    // Host:Port、TLS 选择
    char        _hostname[256] = {0};
    char        _port[32]      = {0};
    std::string _tls; // 例如 "rc,ud,sm,self,tcp"

    // 监听到的待接收 EP 队列
    mutable std::mutex              _accept_mtx;
    mutable std::condition_variable _accept_cv;
    mutable std::queue<ucp_ep_h>    _accepted_eps;

    // worker 事件模式（可能不可用）
    int  _worker_event_fd = -1;
    bool _use_event_fd = false;

    // REMOVED: Progress thread members are no longer needed.
    // std::atomic<bool> _progress_run{false};
    // std::thread       _progress_th;

    // 请求状态（挂在 UCX request 的私有区）
    struct _req_state {
        bool         completed;
        ucs_status_t status;
        size_t       bytes;
    };

private:
    // --- 初始化/清理 ---
    void _init_context();
    void _finalize_context();
    void _create_worker();
    void _destroy_worker();
    void _setup_listener();
    void _destroy_listener();

    // --- 事件推进 ---
    // REMOVED: Background thread methods are no longer needed.
    // void _start_progress_thread();
    // void _stop_progress_thread();
    void _progress_once() const;      // 单步 progress
    void _wait_progress() const;      // 事件驱动等待
    void _ensure_progress_until(bool (*pred)(void*), void* arg) const;

    // --- 地址解析 ---
    void _resolve_sockaddr(struct sockaddr_storage& ss, socklen_t& slen) const;

    // --- 回调（必须是 static / C 回调签名） ---
    static void _on_conn_request(ucp_conn_request_h req, void* arg);
    static void _on_ep_error(void* arg, ucp_ep_h ep, ucs_status_t status);
    static void _on_send_complete(void* request, ucs_status_t status, void* user_data);
    static void _on_recv_complete(void* request, ucs_status_t status, size_t length, void* user_data);
    static void _on_ep_close_complete(void* request, ucs_status_t status, void* user_data);

    // --- 阻塞的 Stream 收发 ---
    size_t _blocking_stream_send(const void* buf, size_t size);
    size_t _blocking_stream_recv(void* buf, size_t size);
};

} // namespace communicators
} // namespace gvirtus

// 工厂符号供 dlsym
extern "C"
std::shared_ptr<gvirtus::communicators::UcxCommunicator>
create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> end);