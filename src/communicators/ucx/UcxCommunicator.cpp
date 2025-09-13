#include "UcxCommunicator.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <netdb.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <ifaddrs.h>

#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Ucx.h>

using gvirtus::communicators::UcxCommunicator;

// 辅助宏
namespace {
    inline const char* _ucs_str(ucs_status_t st) { return ucs_status_string(st); }

    #define UCS_THROW_IF_NOT_OK(expr, what) do {                                  \
        ucs_status_t __st = (expr);                                               \
        if (__st != UCS_OK) {                                                     \
            throw std::runtime_error(std::string(what) + ": " + _ucs_str(__st));  \
        }                                                                         \
    } while(0)

    // --- 根据 sockaddr 拿本机 ifname ---
    static std::string ifname_from_sockaddr(const struct sockaddr* sa) {
        struct ifaddrs* ifas = nullptr;
        if (getifaddrs(&ifas) != 0) return "";

        std::string name;
        for (auto* p = ifas; p; p = p->ifa_next) {
            if (!p->ifa_addr) continue;
            if (p->ifa_addr->sa_family != sa->sa_family) continue;

            if (sa->sa_family == AF_INET) {
                auto* a = reinterpret_cast<const sockaddr_in*>(sa);
                auto* b = reinterpret_cast<const sockaddr_in*>(p->ifa_addr);
                if (a->sin_addr.s_addr == b->sin_addr.s_addr) {
                    name = p->ifa_name;
                    break;
                }
            } else if (sa->sa_family == AF_INET6) {
                auto* a = reinterpret_cast<const sockaddr_in6*>(sa);
                auto* b = reinterpret_cast<const sockaddr_in6*>(p->ifa_addr);
                if (memcmp(&a->sin6_addr, &b->sin6_addr, sizeof(in6_addr)) == 0) {
                    name = p->ifa_name;
                    break;
                }
            }
        }
        freeifaddrs(ifas);
        return name;
    }

    // --- 通过路由推导“到对端地址会用哪块本地网卡”并返回 ifname ---
    static std::string route_ifname_to(const struct sockaddr* dst, socklen_t dlen) {
        int s = ::socket(dst->sa_family, SOCK_DGRAM, 0);
        if (s < 0) return "";
        (void)::connect(s, dst, dlen);

        sockaddr_storage local{};
        socklen_t llen = sizeof(local);
        if (getsockname(s, reinterpret_cast<sockaddr*>(&local), &llen) != 0) {
            close(s);
            return "";
        }
        close(s);
        return ifname_from_sockaddr(reinterpret_cast<sockaddr*>(&local));
    }
} // anon namespace for helpers

// ---- UCX stream recv: 准确接收 nbytes ----
// MODIFIED: 成为一个纯粹的字节流接收器，不再关心帧格式
void UcxCommunicator::_recv_stream_exact(ucp_worker_h worker, ucp_ep_h ep, void* buf, size_t nbytes) {
    if (nbytes == 0) return;
    
    size_t total = 0;
    while (total < nbytes) {
        auto req_state = std::make_shared<UcxCommunicator::_req_state>();
        
        ucp_request_param_t p{};
        p.op_attr_mask   = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        p.user_data      = req_state.get();
        p.cb.recv_stream = [](void* request, ucs_status_t status, size_t length, void* user_data) {
            auto* s = reinterpret_cast<UcxCommunicator::_req_state*>(user_data);
            s->signal(status, length);
            ucp_request_free(request);
        };

        size_t got = 0;
        char* current_buf = static_cast<char*>(buf) + total;
        size_t current_len = nbytes - total;

        void* req = ucp_stream_recv_nbx(ep, current_buf, current_len, &got, &p);
        
        if (UCS_PTR_IS_ERR(req)) {
            throw std::runtime_error(std::string("ucp_stream_recv_nbx failed: ") + _ucs_str(UCS_PTR_STATUS(req)));
        }

        if (req == nullptr) {
            total += got;
            continue;
        }
        
        req_state->wait();

        if (req_state->status.load() != UCS_OK) {
            throw std::runtime_error(std::string("stream recv completion: ") + _ucs_str(req_state->status.load()));
        }
        total += req_state->bytes.load();
    }
}

// ---- UCX stream send: 准确发送 nbytes ----
// MODIFIED: 成为一个纯粹的字节流发送器，不添加任何额外数据
void UcxCommunicator::_send_stream_exact(ucp_ep_h ep, const void* buf, size_t nbytes) {
    if (nbytes == 0) return;

    auto req_state = std::make_shared<UcxCommunicator::_req_state>();

    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    p.user_data    = req_state.get();
    p.cb.send = [](void* request, ucs_status_t status, void* user_data) {
        auto* s = reinterpret_cast<UcxCommunicator::_req_state*>(user_data);
        s->signal(status, 0);
        ucp_request_free(request);
    };

    void* req = ucp_stream_send_nbx(ep, buf, nbytes, &p);
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error(std::string("ucp_stream_send_nbx failed: ") + _ucs_str(UCS_PTR_STATUS(req)));
    }
    
    if (req == nullptr) {
        return; // 立即完成
    }

    req_state->wait(); // 等待发送完成
    if (req_state->status.load() != UCS_OK) {
        throw std::runtime_error(std::string("stream send completion: ") + _ucs_str(req_state->status.load()));
    }
}

// ---------------- sockaddr 解析 ----------------
void UcxCommunicator::_resolve_sockaddr(struct sockaddr_storage& ss, socklen_t& slen) const {
    struct addrinfo hints{};
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* res = nullptr;
    int rc = getaddrinfo(_hostname, _port, &hints, &res);
    if (rc != 0 || !res) {
        std::ostringstream oss;
        oss << "UcxCommunicator: getaddrinfo failed for "
            << _hostname << ":" << _port << " (" << gai_strerror(rc) << ")";
        throw std::runtime_error(oss.str());
    }
    memcpy(&ss, res->ai_addr, res->ai_addrlen);
    slen = static_cast<socklen_t>(res->ai_addrlen);
    freeaddrinfo(res);
}

// ---------------- 上下文/工作器 ----------------
void UcxCommunicator::_init_context() {
    ucp_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask =
        UCP_PARAM_FIELD_FEATURES |
        UCP_PARAM_FIELD_NAME;
    params.features = UCP_FEATURE_STREAM | UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
    params.name = "gvirtus_ucx";

    ucp_config_t* cfg = nullptr;
    UCS_THROW_IF_NOT_OK(ucp_config_read(nullptr, nullptr, &cfg), "ucp_config_read");

    if (!_bind_ifname.empty()) {
        ucp_config_modify(cfg, "NET_DEVICES", _bind_ifname.c_str());
        ucp_config_modify(cfg, "SOCKADDR_IFNAME", _bind_ifname.c_str());
    } else if (const char* nd = std::getenv("UCX_NET_DEVICES")) {
        ucp_config_modify(cfg, "NET_DEVICES", nd);
        ucp_config_modify(cfg, "SOCKADDR_IFNAME", nd);
    }

    ucp_config_modify(cfg, "SOCKADDR_TLS_PRIORITY", "tcp");
    ucp_config_modify(cfg, "SOCKADDR_AUX_TLS",      "none");

    if (!_tls.empty()) {
        ucp_config_modify(cfg, "TLS", _tls.c_str());
    }

    UCS_THROW_IF_NOT_OK(ucp_init(&params, cfg, &_context), "ucp_init");
    ucp_config_release(cfg);
}

void UcxCommunicator::_finalize_context() {
    if (_context) {
        ucp_cleanup(_context);
        _context = nullptr;
    }
}

void UcxCommunicator::_create_worker() {
    ucp_worker_params_t wparams;
    memset(&wparams, 0, sizeof(wparams));
    wparams.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    wparams.thread_mode = UCS_THREAD_MODE_MULTI;

    UCS_THROW_IF_NOT_OK(ucp_worker_create(_context, &wparams, &_worker), "ucp_worker_create");

    int efd = -1;
    if (ucp_worker_get_efd(_worker, &efd) == UCS_OK) {
        _worker_event_fd = efd;
        _use_event_fd    = true;
    } else {
        _worker_event_fd = -1;
        _use_event_fd    = false;
    }
}

void UcxCommunicator::_destroy_worker() {
    if (_worker) {
        ucp_worker_destroy(_worker);
        _worker = nullptr;
        _worker_event_fd = -1;
        _use_event_fd = false;
    }
}

// ---------------- 监听 ----------------
void UcxCommunicator::_setup_listener() {
    ucp_listener_params_t lp;
    memset(&lp, 0, sizeof(lp));
    lp.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                    UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    lp.sockaddr.addr    = reinterpret_cast<const struct sockaddr*>(&_listen_addr);
    lp.sockaddr.addrlen = _listen_addrlen;
    lp.conn_handler.cb  = &UcxCommunicator::_on_conn_request;
    lp.conn_handler.arg = this;

    UCS_THROW_IF_NOT_OK(ucp_listener_create(_worker, &lp, &_listener), "ucp_listener_create");
}

void UcxCommunicator::_destroy_listener() {
    if (_listener) {
        ucp_listener_destroy(_listener);
        _listener = nullptr;
    }
}

// ---------------- 进度处理 ----------------
void UcxCommunicator::_progress_once() const {
    if (_worker) ucp_worker_progress(_worker);
}

void UcxCommunicator::_wait_progress() const {
    if (!_worker) return;

    ucs_status_t st = ucp_worker_arm(_worker);
    if (st == UCS_ERR_BUSY) {
    } else if (st == UCS_OK) {
        ucp_worker_wait(_worker);
        return;
    }
    ucp_worker_progress(_worker);
}

void UcxCommunicator::ProgressOnce() { _progress_once(); }

void UcxCommunicator::StartProgress() {
    std::lock_guard<std::mutex> lk(_progress_mtx);
    if (_progress_running) return;
    _progress_running = true;
    _progress_thread = std::thread([this](){
        while (_progress_running && !_closing) {
            if (ucp_worker_progress(_worker) == 0) {
                _wait_progress();
            }
        }
    });
}

void UcxCommunicator::StopProgress() {
    {
        std::lock_guard<std::mutex> lk(_progress_mtx);
        if (!_progress_running) return;
        _progress_running = false;
    }
    if (_worker) {
        ucp_worker_signal(_worker);
    }
    if (_progress_thread.joinable()) _progress_thread.join();
}

// ---------------- 回调 ----------------
void UcxCommunicator::_on_conn_request(ucp_conn_request_h req, void* arg) {
    auto* self = reinterpret_cast<UcxCommunicator*>(arg);

    ucp_ep_params_t ep_params{};
    ep_params.field_mask =
        UCP_EP_PARAM_FIELD_CONN_REQUEST |
        UCP_EP_PARAM_FIELD_ERR_HANDLER |
        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
        UCP_EP_PARAM_FIELD_USER_DATA;
    ep_params.conn_request    = req;
    ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb  = &UcxCommunicator::_on_ep_error;
    ep_params.err_handler.arg = self;
    ep_params.user_data       = self;

    ucp_ep_h ep = nullptr;
    ucs_status_t st = ucp_ep_create(self->_worker, &ep_params, &ep);
    if (st != UCS_OK) {
        #ifdef DEBUG
        std::cerr << "ucp_ep_create on accept failed: " << _ucs_str(st) << std::endl;
        #endif
        return;
    }

    {
        std::lock_guard<std::mutex> lk(self->_accept_mtx);
        self->_accepted_eps.push(ep);
    }
    self->_accept_cv.notify_one();
}

void UcxCommunicator::_on_ep_error(void* arg, ucp_ep_h /*ep*/, ucs_status_t status) {
    #ifdef DEBUG
    (void)arg;
    std::cerr << "UCX endpoint error: " << _ucs_str(status) << std::endl;
    #else
    (void)arg; (void)status;
    #endif
}

// ---------------- 异步通信核心 (已不再需要，由阻塞版本直接实现) ----------------
std::shared_ptr<UcxCommunicator::AsyncRequest>
UcxCommunicator::WriteAsync(const void* buf, size_t size) {
    // 保持API兼容性，但在内部实现为阻塞写
    Write(static_cast<const char*>(buf), size);
    
    auto areq = std::make_shared<AsyncRequest>();
    areq->status.store(UCS_OK);
    areq->bytes.store(size);
    areq->completed.store(true);
    if (areq->on_complete) areq->on_complete(UCS_OK, size);
    return areq;
}

std::shared_ptr<UcxCommunicator::AsyncRequest>
UcxCommunicator::ReadAsync(void* buf, size_t max_size) {
    // 保持API兼容性，但在内部实现为阻塞读
    size_t bytes_read = Read(static_cast<char*>(buf), max_size);
    
    auto areq = std::make_shared<AsyncRequest>();
    areq->status.store(UCS_OK); // 假设成功，因为如果失败Read会抛异常
    areq->bytes.store(bytes_read);
    areq->completed.store(true);
    if (areq->on_complete) areq->on_complete(UCS_OK, bytes_read);
    return areq;
}

// ---------------- 构造/析构 ----------------
UcxCommunicator::UcxCommunicator(const std::string& hostname, const std::string& port)
: _tls("") {
    strncpy(_hostname, hostname.c_str(), sizeof(_hostname)-1);
    _hostname[sizeof(_hostname)-1] = '\0';
    strncpy(_port,     port.c_str(),     sizeof(_port)-1);
    _port[sizeof(_port)-1] = '\0';
}

UcxCommunicator::UcxCommunicator(const std::string& hostname, const std::string& port, const std::string& tls)
: _tls(tls) {
    strncpy(_hostname, hostname.c_str(), sizeof(_hostname)-1);
    _hostname[sizeof(_hostname)-1] = '\0';
    strncpy(_port,     port.c_str(),     sizeof(_port)-1);
    _port[sizeof(_port)-1] = '\0';
}

UcxCommunicator::UcxCommunicator(ucp_context_h context, ucp_worker_h worker, ucp_ep_h ep) {
    _context = context;
    _worker  = worker;
    _ep      = ep;
    _is_server_side_wrapper = true;
}

UcxCommunicator::~UcxCommunicator() {
    try { Close(); } catch (...) { }
}

// ---------------- 外部 API：监听/连接 ----------------
void UcxCommunicator::Serve() {
    _resolve_sockaddr(_listen_addr, _listen_addrlen);
    _bind_ifname = ifname_from_sockaddr(reinterpret_cast<const sockaddr*>(&_listen_addr));

    if (_context == nullptr) _init_context();
    if (_worker  == nullptr) _create_worker();

    _setup_listener();
    StartProgress();
}

const gvirtus::communicators::Communicator* const UcxCommunicator::Accept() const {
    ucp_ep_h new_ep = nullptr;

    std::unique_lock<std::mutex> lk(_accept_mtx);
    _accept_cv.wait(lk, [this]{ return !_accepted_eps.empty() || _closing; });

    if (_closing && _accepted_eps.empty()) {
        return nullptr;
    }
    
    new_ep = _accepted_eps.front();
    const_cast<std::queue<ucp_ep_h>&>(_accepted_eps).pop();

    return new UcxCommunicator(_context, _worker, new_ep);
}

void UcxCommunicator::Connect() {
    struct sockaddr_storage ss{};
    socklen_t slen = 0;
    _resolve_sockaddr(ss, slen);
    _bind_ifname = route_ifname_to(reinterpret_cast<const sockaddr*>(&ss), slen);

    if (_context == nullptr) _init_context();
    if (_worker  == nullptr) _create_worker();

    ucp_ep_params_t ep_params{};
    ep_params.field_mask =
        UCP_EP_PARAM_FIELD_SOCK_ADDR |
        UCP_EP_PARAM_FIELD_ERR_HANDLER |
        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
        UCP_EP_PARAM_FIELD_USER_DATA |
        UCP_EP_PARAM_FIELD_FLAGS;

    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = &UcxCommunicator::_on_ep_error;
    ep_params.err_handler.arg  = this;
    ep_params.user_data        = this;
    ep_params.sockaddr.addr    = reinterpret_cast<const sockaddr*>(&ss);
    ep_params.sockaddr.addrlen = slen;

    UCS_THROW_IF_NOT_OK(ucp_ep_create(_worker, &ep_params, &_ep), "ucp_ep_create");
    
    StartProgress();
}

// ---------------- 阻塞封装 ----------------
// MODIFIED: Write 现在是一个纯粹的字节流发送器
size_t UcxCommunicator::Write(const char* buffer, size_t size) {
    if (!_ep) throw std::runtime_error("Write on null endpoint");
    _send_stream_exact(_ep, buffer, size);
    return size;
}

// MODIFIED: Read 现在是一个纯粹的字节流接收器
size_t UcxCommunicator::Read(char* buffer, size_t size) {
    if (!_ep) throw std::runtime_error("Read on null endpoint");
    _recv_stream_exact(_worker, _ep, buffer, size);
    return size;
}
std::string UcxCommunicator::to_string() { // 移除 const
    return "ucxcommunicator";
}
void UcxCommunicator::Sync() {
    if (!_ep) return;
    
    ucp_request_param_t p{};
    void* req = ucp_ep_flush_nbx(_ep, &p);

    if (req == nullptr) return;
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error(std::string("ucp_ep_flush_nbx failed: ") + _ucs_str(UCS_PTR_STATUS(req)));
    }

    // 在等待flush时，我们不能阻塞整个进度线程，所以在这里自己驱动一下
    while (ucp_request_check_status(req) == UCS_INPROGRESS) {
        ucp_worker_progress(_worker);
    }
    
    ucs_status_t status = ucp_request_check_status(req);
    ucp_request_free(req);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Sync flush completion: ") + _ucs_str(status));
    }
}

void UcxCommunicator::Close() {
    if (_closing.exchange(true)) return;

    _accept_cv.notify_all(); 

    StopProgress();

    if (_ep) {
        ucp_request_param_t params{};
        params.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
        params.flags        = UCP_EP_CLOSE_FLAG_FORCE;
        void* req = ucp_ep_close_nbx(_ep, &params);
        
        if (req != nullptr && !UCS_PTR_IS_ERR(req)) {
            // 在关闭时，短暂地自己驱动进度来完成关闭请求
            while (ucp_request_check_status(req) == UCS_INPROGRESS) {
                if (_worker) ucp_worker_progress(_worker);
            }
            ucp_request_free(req);
        }
        _ep = nullptr;
    }

    if (!_is_server_side_wrapper) {
        _destroy_listener();
        _destroy_worker();
        _finalize_context();
    }
}

// ---------------- 工厂 ----------------
extern "C"
std::shared_ptr<gvirtus::communicators::UcxCommunicator>
create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    auto ep = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
    if (!ep) {
        throw std::runtime_error("UCX create_communicator: bad endpoint type (expected Endpoint_Ucx)");
    }
    const std::string hostname = ep->address();
    const std::string port     = std::to_string(ep->port());
    return std::make_shared<gvirtus::communicators::UcxCommunicator>(hostname, port);
}