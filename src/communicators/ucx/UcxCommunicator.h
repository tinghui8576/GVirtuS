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

#include <ucp/api/ucp.h>
#include <gvirtus/communicators/Communicator.h>

namespace gvirtus {
namespace communicators {

class UcxCommunicator : public Communicator {
public:
    // 构造/析构
    UcxCommunicator(const std::string& hostname, const std::string& port);
    UcxCommunicator(const std::string& hostname, const std::string& port, const std::string& tls);
    UcxCommunicator(ucp_context_h context, ucp_worker_h worker, ucp_ep_h ep);
    ~UcxCommunicator() override;

    // gvirtus 接口（阻塞式封装）
    void   Serve() override;
    void   Connect() override;
    size_t Read(char* buffer, size_t size) override;
    size_t Write(const char* buffer, size_t size) override;
    void   Sync() override;
    void   Close() override;
    const Communicator* const Accept() const override;

    // -------- 新增：异步接口 --------
    struct AsyncRequest {
        std::atomic<bool> completed{false};
        std::atomic<ucs_status_t> status{UCS_INPROGRESS};
        std::atomic<size_t> bytes{0};
        std::function<void(ucs_status_t,size_t)> on_complete;
    };

    std::shared_ptr<AsyncRequest> WriteAsync(const void* buf, size_t size);
    std::shared_ptr<AsyncRequest> ReadAsync(void* buf, size_t max_size);
        // UCX request 私有区
    struct _req_state {
        bool         completed;
        ucs_status_t status;
        size_t       bytes;
    };

    // 进度
    void ProgressOnce();
    void StartProgress();
    void StopProgress();

private:
    // UCX 基本对象
    ucp_context_h  _context = nullptr;
    ucp_worker_h   _worker  = nullptr;
    ucp_listener_h _listener = nullptr;
    ucp_ep_h       _ep = nullptr;

    // 状态
    bool _is_server_side_wrapper = false;
    bool _closing = false;

    char        _hostname[256] = {0};
    char        _port[32]      = {0};
    std::string _tls;

    // accept 队列
    mutable std::mutex              _accept_mtx;
    mutable std::condition_variable _accept_cv;
    mutable std::queue<ucp_ep_h>    _accepted_eps;

    int  _worker_event_fd = -1;
    bool _use_event_fd = false;

    // progress 线程
    std::atomic<bool> _progress_running{false};
    std::thread       _progress_thread;
    std::mutex        _progress_mtx;


    // 内部工具
    void _init_context();
    void _finalize_context();
    void _create_worker();
    void _destroy_worker();
    void _setup_listener();
    void _destroy_listener();
    void _progress_once() const;
    void _wait_progress() const;
    void _ensure_progress_until(bool (*pred)(void*), void* arg) const;
    void _resolve_sockaddr(struct sockaddr_storage& ss, socklen_t& slen) const;

    // 回调
    static void _on_conn_request(ucp_conn_request_h req, void* arg);
    static void _on_ep_error(void* arg, ucp_ep_h ep, ucs_status_t status);
    static void _on_send_complete(void* request, ucs_status_t status, void* user_data);
    static void _on_recv_complete(void* request, ucs_status_t status, size_t length, void* user_data);
    static void _on_ep_close_complete(void* request, ucs_status_t status, void* user_data);

    // 阻塞实现（内部用，供同步 API）
    size_t _blocking_stream_send(const void* buf, size_t size);
    size_t _blocking_stream_recv(void* buf, size_t size);
};

} // namespace communicators
} // namespace gvirtus

extern "C"
std::shared_ptr<gvirtus::communicators::UcxCommunicator>
create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> end);
