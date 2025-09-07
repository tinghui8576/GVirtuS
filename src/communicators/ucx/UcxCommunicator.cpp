#include "UcxCommunicator.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <netdb.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Ucx.h>

using gvirtus::communicators::UcxCommunicator;

namespace {
inline const char* _ucs_str(ucs_status_t st) { return ucs_status_string(st); }
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
        UCP_PARAM_FIELD_REQUEST_SIZE |
        UCP_PARAM_FIELD_REQUEST_INIT |
        UCP_PARAM_FIELD_NAME;
    params.features = UCP_FEATURE_STREAM | UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;

    params.request_size = sizeof(_req_state);
    params.request_init = [](void* req) {
        auto* s = reinterpret_cast<_req_state*>(req);
        s->completed = false;
        s->status    = UCS_INPROGRESS;
        s->bytes     = 0;
    };
    params.name = "gvirtus_ucx";

    ucp_config_t* cfg = nullptr;
    ucs_status_t st = ucp_config_read(nullptr, nullptr, &cfg);
    if (st != UCS_OK) throw std::runtime_error(std::string("ucp_config_read failed: ") + _ucs_str(st));

    // 统一优先 TCP 作为 sockaddr CM；TLS 同时允许 IB + TCP（满足你“tcp+ib”双栈需求）
    (void)ucp_config_modify(cfg, "SOCKADDR_TLS_PRIORITY", "tcp");
    if (!_tls.empty()) {
        (void)ucp_config_modify(cfg, "TLS", _tls.c_str());
    } else {
        // 默认既开 IB（rc/ud）也开 TCP、自环/共享内存
        (void)ucp_config_modify(cfg, "TLS", "rc,ud,sm,self,tcp");
    }

    st = ucp_init(&params, cfg, &_context);
    ucp_config_release(cfg);
    if (st != UCS_OK) throw std::runtime_error(std::string("ucp_init failed: ") + _ucs_str(st));
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
    wparams.thread_mode = UCS_THREAD_MODE_SINGLE;

    ucs_status_t st = ucp_worker_create(_context, &wparams, &_worker);
    if (st != UCS_OK) throw std::runtime_error("ucp_worker_create failed");

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

void UcxCommunicator::_setup_listener() {
    struct sockaddr_storage ss{};
    socklen_t slen = 0;
    _resolve_sockaddr(ss, slen);

    ucp_listener_params_t lp;
    memset(&lp, 0, sizeof(lp));
    lp.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                    UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    lp.sockaddr.addr    = reinterpret_cast<const struct sockaddr*>(&ss);
    lp.sockaddr.addrlen = slen;
    lp.conn_handler.cb  = &UcxCommunicator::_on_conn_request;
    lp.conn_handler.arg = this;

    ucs_status_t st = ucp_listener_create(_worker, &lp, &_listener);
    if (st != UCS_OK) throw std::runtime_error("ucp_listener_create failed");
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

    // 事件驱动优先，否则 busy progress
    ucs_status_t st = ucp_worker_arm(_worker);
    if (st == UCS_ERR_BUSY) {
        return; // 已有事件，下一轮 progress 会处理
    } else if (st == UCS_OK) {
        ucp_worker_wait(_worker); // 旧版 API
        return;
    }
    // arm 失败就主动推进一次
    ucp_worker_progress(_worker);
}

void UcxCommunicator::_ensure_progress_until(bool (*pred)(void*), void* arg) const {
    while (!pred(arg)) {
        if (ucp_worker_progress(_worker) == 0) {
            _wait_progress();
        }
    }
}

void UcxCommunicator::_start_progress_thread() {
    bool expected = false;
    if (!_progress_run.compare_exchange_strong(expected, true)) return; // 已经在跑

    _progress_th = std::thread([this] {
        // 轻量推进循环
        while (_progress_run.load(std::memory_order_relaxed)) {
            if (_worker && (ucp_worker_progress(_worker) == 0)) {
                _wait_progress();
            }
        }
    });
}

void UcxCommunicator::_stop_progress_thread() {
    if (_progress_run.exchange(false)) {
        if (_progress_th.joinable()) _progress_th.join();
    }
}

// ---------------- 回调 ----------------
void UcxCommunicator::_on_conn_request(ucp_conn_request_h req, void* arg) {
    auto* self = reinterpret_cast<UcxCommunicator*>(arg);

    // UCX 要求在 on_conn_request 里创建 server-side endpoint
    ucp_ep_params_t ep_params{};
    ep_params.field_mask =
        UCP_EP_PARAM_FIELD_CONN_REQUEST |
        UCP_EP_PARAM_FIELD_ERR_HANDLER |
        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
        UCP_EP_PARAM_FIELD_USER_DATA;
    ep_params.conn_request   = req;
    ep_params.err_mode       = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb = &UcxCommunicator::_on_ep_error;
    ep_params.err_handler.arg = self;
    ep_params.user_data      = self;

    ucp_ep_h ep = nullptr;
    ucs_status_t st = ucp_ep_create(self->_worker, &ep_params, &ep);
    if (st != UCS_OK) {
        // 如果失败，也可以考虑 ucp_listener_reject(req)；此处直接返回。
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
    auto* self = reinterpret_cast<UcxCommunicator*>(arg);
    (void)self;
    std::cerr << "UCX endpoint error: " << _ucs_str(status) << std::endl;
#else
    (void)arg; (void)status;
#endif
}

void UcxCommunicator::_on_send_complete(void* request, ucs_status_t status, void* /*user_data*/) {
    auto* s = reinterpret_cast<_req_state*>(request);
    s->status    = status;
    s->completed = true;
}

void UcxCommunicator::_on_recv_complete(void* request, ucs_status_t status, size_t length, void* /*user_data*/) {
    auto* s = reinterpret_cast<_req_state*>(request);
    s->status    = status;
    s->bytes     = length;
    s->completed = true;
}

void UcxCommunicator::_on_ep_close_complete(void* request, ucs_status_t status, void* /*user_data*/) {
    auto* s = reinterpret_cast<_req_state*>(request);
    s->status    = status;
    s->completed = true;
}

// ---------------- 阻塞 STREAM 收发 ----------------
size_t UcxCommunicator::_blocking_stream_send(const void* buf, size_t size) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    p.cb.send      = &UcxCommunicator::_on_send_complete;

    void* req = ucp_stream_send_nbx(_ep, buf, size, &p);
    if (req == nullptr) return size;
    if (UCS_PTR_IS_ERR(req)) throw std::runtime_error(std::string("ucp_stream_send_nbx failed: ") + _ucs_str(UCS_PTR_STATUS(req)));

    auto* s = reinterpret_cast<_req_state*>(req);
    auto pred = [](void* arg)->bool { return reinterpret_cast<_req_state*>(arg)->completed; };
    _ensure_progress_until(pred, s);

    ucs_status_t st = ucp_request_check_status(req);
    ucp_request_free(req);
    if (st != UCS_OK) throw std::runtime_error(std::string("stream send completion: ") + _ucs_str(st));
    return size;
}

size_t UcxCommunicator::_blocking_stream_recv(void* buf, size_t size) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    p.op_attr_mask   = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    p.cb.recv_stream = &UcxCommunicator::_on_recv_complete;

    size_t got = 0;
    void* req = ucp_stream_recv_nbx(_ep, buf, size, &got, &p);
    if (req == nullptr) {
        return got; // 立即完成
    }
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error(std::string("ucp_stream_recv_nbx failed: ")
                                 + _ucs_str(UCS_PTR_STATUS(req)));
    }

    auto* s = reinterpret_cast<_req_state*>(req);
    auto pred = [](void* arg)->bool { return reinterpret_cast<_req_state*>(arg)->completed; };
    _ensure_progress_until(pred, s);

    ucs_status_t st = ucp_request_check_status(req);
    size_t rx = s->bytes;
    ucp_request_free(req);
    if (st != UCS_OK) {
        throw std::runtime_error(std::string("stream recv completion: ") + _ucs_str(st));
    }
    return rx;
}

// ---------------- 构造/析构 ----------------
UcxCommunicator::UcxCommunicator(const std::string& hostname, const std::string& port)
: _tls("rc,ud,sm,self,tcp") {
    strncpy(_hostname, hostname.c_str(), sizeof(_hostname)-1);
    strncpy(_port,     port.c_str(),     sizeof(_port)-1);
}

UcxCommunicator::UcxCommunicator(const std::string& hostname, const std::string& port, const std::string& tls)
: _tls(tls) {
    strncpy(_hostname, hostname.c_str(), sizeof(_hostname)-1);
    strncpy(_port,     port.c_str(),     sizeof(_port)-1);
}

UcxCommunicator::UcxCommunicator(ucp_context_h context, ucp_worker_h worker, ucp_ep_h ep) {
    _context = context;
    _worker  = worker;
    _ep      = ep;
    _is_server_side_wrapper = true; // 只包一层 EP，不负责清理 context/worker
}

UcxCommunicator::~UcxCommunicator() {
    try { Close(); } catch (...) {}
    if (!_is_server_side_wrapper) {
        _stop_progress_thread();
        _destroy_listener();
        _destroy_worker();
        _finalize_context();
    }
}

// ---------------- 外部 API ----------------
void UcxCommunicator::Serve() {
    if (_context == nullptr) _init_context();
    if (_worker  == nullptr) _create_worker();
    _setup_listener();
}

const gvirtus::communicators::Communicator* const UcxCommunicator::Accept() const {
    ucp_ep_h new_ep = nullptr;

    for (;;) {
        {   // 先看看队列里有没有
            std::unique_lock<std::mutex> lk(_accept_mtx);
            if (!_accepted_eps.empty()) {
                new_ep = _accepted_eps.front();
                const_cast<std::queue<ucp_ep_h>&>(_accepted_eps).pop();
                break;
            }
        }
        // 没有的话，推进一下；若没事件，再走唤醒等待
        if (ucp_worker_progress(_worker) == 0) {
            ucs_status_t st = ucp_worker_arm(_worker);
            if (st == UCS_OK) {
                ucp_worker_wait(_worker);  // 被唤醒后，再回到循环顶部
            } else if (st != UCS_ERR_BUSY) {
                // arm 失败（非 BUSY），退化为主动推进
                sched_yield();
            }
        }
    }

    return new UcxCommunicator(_context, _worker, new_ep);
}

void UcxCommunicator::Connect() {
    if (_context == nullptr) _init_context();
    if (_worker  == nullptr) _create_worker();

    struct sockaddr_storage ss{};
    socklen_t slen = 0;
    _resolve_sockaddr(ss, slen);
    if (slen == 0) {
        throw std::runtime_error("UcxCommunicator::Connect(): resolve_sockaddr returned len=0");
    }

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

    ucs_status_t st = ucp_ep_create(_worker, &ep_params, &_ep);
    if (st != UCS_OK) {
        throw std::runtime_error("ucp_ep_create failed: " + std::string(_ucs_str(st)));
    }


    // 不要立刻 flush；等有首个消息再 Sync() 或由上层协议握手
}

size_t UcxCommunicator::Read(char* buffer, size_t size) {
    if (!_ep) throw std::runtime_error("Read on null endpoint");
    return _blocking_stream_recv(buffer, size);
}

size_t UcxCommunicator::Write(const char* buffer, size_t size) {
    if (!_ep) throw std::runtime_error("Write on null endpoint");
    return _blocking_stream_send(buffer, size);
}

void UcxCommunicator::Sync() {
    if (!_ep) return;

    ucp_request_param_t p;
    memset(&p, 0, sizeof(p)); // 不设回调

    void* req = ucp_ep_flush_nbx(_ep, &p);
    if (req == nullptr) return;                    // 立即完成
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error(std::string("ucp_ep_flush_nbx failed: ")
                                 + _ucs_str(UCS_PTR_STATUS(req)));
    }

    for (;;) {
        ucs_status_t st = ucp_request_check_status(req);
        if (st == UCS_OK) break;
        if (st != UCS_INPROGRESS) {
            ucp_request_free(req);
            throw std::runtime_error(std::string("Sync flush completion: ")
                                     + _ucs_str(st));
        }
        // 单线程推进：没事件就 arm+wait，有事件就 progress
        if (ucp_worker_progress(_worker) == 0) {
            ucs_status_t a = ucp_worker_arm(_worker);
            if (a == UCS_OK) {
                ucp_worker_wait(_worker);
            } else if (a != UCS_ERR_BUSY) {
                sched_yield();
            }
        }
    }

    ucp_request_free(req);
}

void UcxCommunicator::Close() {
    if (_closing) return;
    _closing = true;

    if (_ep) {
        ucp_request_param_t p;
        memset(&p, 0, sizeof(p));
        p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        p.cb.send      = &UcxCommunicator::_on_ep_close_complete;

        void* req = ucp_ep_close_nbx(_ep, &p);
        if (UCS_PTR_IS_PTR(req)) {
            auto* s = reinterpret_cast<_req_state*>(req);
            auto pred = [](void* arg)->bool { return reinterpret_cast<_req_state*>(arg)->completed; };
            _ensure_progress_until(pred, s);
            ucp_request_free(req);
        }
        _ep = nullptr;
    }

    // 仅主对象（非 server-side wrapper）才停线程/销毁 listener/worker
    if (!_is_server_side_wrapper) {
        _stop_progress_thread();
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
