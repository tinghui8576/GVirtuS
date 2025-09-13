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
    UcxCommunicator(const std::string& hostname, const std::string& port);
    UcxCommunicator(const std::string& hostname, const std::string& port, const std::string& tls);
    UcxCommunicator(ucp_context_h context, ucp_worker_h worker, ucp_ep_h ep);
    ~UcxCommunicator() override;

    void   Serve() override;
    void   Connect() override;
    size_t Read(char* buffer, size_t size) override;
    size_t Write(const char* buffer, size_t size) override;
    void   Sync() override;
    void   Close() override;
    const Communicator* const Accept() const override;
    std::string to_string() override;

    // 异步接口 (为保持API兼容性而保留，但内部实现为阻塞)
    struct AsyncRequest {
        std::atomic<bool> completed{false};
        std::atomic<ucs_status_t> status{UCS_INPROGRESS};
        std::atomic<size_t> bytes{0};
        std::function<void(ucs_status_t,size_t)> on_complete;
    };
    std::shared_ptr<AsyncRequest> WriteAsync(const void* buf, size_t size);
    std::shared_ptr<AsyncRequest> ReadAsync(void* buf, size_t max_size);

    void ProgressOnce();
    void StartProgress();
    void StopProgress();
private:
    struct _req_state {
        std::atomic<bool>           completed{false};
        std::atomic<ucs_status_t>   status{UCS_INPROGRESS};
        std::atomic<size_t>         bytes{0};
        std::mutex mtx;
        std::condition_variable cv;

        void wait() {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [this]{ return completed.load(); });
        }
        void signal(ucs_status_t st, size_t b) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                status.store(st);
                bytes.store(b);
                completed.store(true);
            }
            cv.notify_one();
        }
    };

    // UCX 对象
    ucp_context_h  _context  = nullptr;
    ucp_worker_h   _worker   = nullptr;
    ucp_listener_h _listener = nullptr;
    ucp_ep_h       _ep       = nullptr;

    // 状态
    bool _is_server_side_wrapper = false;
    std::atomic<bool> _closing{false};
    char        _hostname[256] = {0};
    char        _port[32]      = {0};
    std::string _tls;

    // listener 相关
    struct sockaddr_storage _listen_addr{};
    socklen_t               _listen_addrlen = 0;
    std::string _bind_ifname;
    mutable std::mutex              _accept_mtx;
    mutable std::condition_variable _accept_cv;
    mutable std::queue<ucp_ep_h>    _accepted_eps;

    // worker 相关
    int  _worker_event_fd = -1;
    bool _use_event_fd    = false;
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
    void _resolve_sockaddr(struct sockaddr_storage& ss, socklen_t& slen) const;
    void _recv_stream_exact(ucp_worker_h worker, ucp_ep_h ep, void* buf, size_t nbytes);
    void _send_stream_exact(ucp_ep_h ep, const void* buf, size_t nbytes);

    // UCX 回调
    static void _on_conn_request(ucp_conn_request_h req, void* arg);
    static void _on_ep_error(void* arg, ucp_ep_h ep, ucs_status_t status);
};

} // namespace communicators
} // namespace gvirtus

extern "C"
std::shared_ptr<gvirtus::communicators::UcxCommunicator>
create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> end);