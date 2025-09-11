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

// 简单工具宏，统一报错
#define UCS_THROW_IF_NOT_OK(expr, what) do { \
    ucs_status_t __st = (expr);               \
    if (__st != UCS_OK) {                     \
        throw std::runtime_error(std::string(what) + ": " + _ucs_str(__st)); \
    }                                         \
} while(0)

// 流分帧：4B 长度头（小端/主机序一致即可，双方一致就行）
#pragma pack(push, 1)
struct FrameHeader { uint32_t len; };
#pragma pack(pop)

} // anon

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
    UCS_THROW_IF_NOT_OK(ucp_config_read(nullptr, nullptr, &cfg), "ucp_config_read");

    // 关键修正：SOCKADDR_TLS_PRIORITY 应该是 rdmacm/sockcm（连接管理层）
    ucp_config_modify(cfg, "SOCKADDR_TLS_PRIORITY", "rdmacm,sockcm");

    // 数据面 TLS
    if (!_tls.empty()) {
        ucp_config_modify(cfg, "TLS", _tls.c_str());
    } else {
        // 你可按实际 UCX 构建选择 rc_x, dc_x；保持兼容性用 rc,ud,sm,self,tcp
        ucp_config_modify(cfg, "TLS", "rc,ud,sm,self,tcp");
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

    UCS_THROW_IF_NOT_OK(ucp_listener_create(_worker, &lp, &_listener), "ucp_listener_create");
}

void UcxCommunicator::_destroy_listener() {
    if (_listener) {
        ucp_listener_destroy(_listener);
        _listener = nullptr;
    }
}

// ---------------- 进度处理（支持进度线程） ----------------
void UcxCommunicator::_progress_once() const {
    if (_worker) ucp_worker_progress(_worker);
}

void UcxCommunicator::_wait_progress() const {
    if (!_worker) return;

    ucs_status_t st = ucp_worker_arm(_worker);
    if (st == UCS_ERR_BUSY) {
        // Already has events pending; fall back to progress
    } else if (st == UCS_OK) {
        ucp_worker_wait(_worker); // event-driven wait
        return;
    }
    ucp_worker_progress(_worker);
}

void UcxCommunicator::ProgressOnce() { _progress_once(); }

void UcxCommunicator::_ensure_progress_until(bool (*pred)(void*), void* arg) const {
    while (!pred(arg)) {
        if (ucp_worker_progress(_worker) == 0) {
            _wait_progress();
        }
    }
}

// 进度线程
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
    ep_params.conn_request   = req;
    ep_params.err_mode       = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb = &UcxCommunicator::_on_ep_error;
    ep_params.err_handler.arg = self;
    ep_params.user_data      = self;

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

// ---------------- 异步请求：公共回调封装 ----------------
static void _mark_done(void* request, ucs_status_t status, size_t bytes = 0) {
    auto* s = reinterpret_cast<UcxCommunicator::_req_state*>(request);
    s->status    = status;
    s->bytes     = bytes;
    s->completed = true;
}

// ---------------- 构造/析构 ----------------
UcxCommunicator::UcxCommunicator(const std::string& hostname, const std::string& port)
: _tls("rc,ud,sm,self,tcp") {
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
    try { Close(); } catch (...) { /* Suppress exceptions in destructor */ }
    if (!_is_server_side_wrapper) {
        _destroy_listener();
        _destroy_worker();
        _finalize_context();
    }
}

// ---------------- 外部 API：监听/连接 ----------------
void UcxCommunicator::Serve() {
    if (_context == nullptr) _init_context();
    if (_worker  == nullptr) _create_worker();
    _setup_listener();
    StartProgress(); // 建议：服务端默认起进度线程
}

const gvirtus::communicators::Communicator* const UcxCommunicator::Accept() const {
    ucp_ep_h new_ep = nullptr;

    std::unique_lock<std::mutex> lk(_accept_mtx);
    for (;;) {
        if (!_accepted_eps.empty()) {
            new_ep = _accepted_eps.front();
            const_cast<std::queue<ucp_ep_h>&>(_accepted_eps).pop();
            break;
        }
        lk.unlock();
        if (ucp_worker_arm(_worker) == UCS_OK) {
            ucp_worker_wait(_worker);
        } else {
            ucp_worker_progress(_worker);
            // 轻微休眠/等待条件变量，避免忙等
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        lk.lock();
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

    UCS_THROW_IF_NOT_OK(ucp_ep_create(_worker, &ep_params, &_ep), "ucp_ep_create");
    StartProgress(); // 建议：客户端也默认起进度线程
}

// ---------------- 异步通信核心 ----------------

// 内部：提交一个非阻塞 stream send，回调里只更新 req_state
static void* _post_stream_send(ucp_ep_h ep, const void* buf, size_t len) {
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    p.cb.send = [](void* request, ucs_status_t status, void*) {
        _mark_done(request, status, 0);
    };

    void* req = ucp_stream_send_nbx(ep, buf, len, &p);
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error(std::string("ucp_stream_send_nbx failed: ") + _ucs_str(UCS_PTR_STATUS(req)));
    }
    return req; // nullptr 代表立即完成；指针代表异步
}

// 内部：提交一个非阻塞 stream recv，回调里记录 bytes
static void* _post_stream_recv(ucp_ep_h ep, void* buf, size_t len) {
    size_t got = 0;
    ucp_request_param_t p{};
    p.op_attr_mask   = UCP_OP_ATTR_FIELD_CALLBACK;
    p.cb.recv_stream = [](void* request, ucs_status_t status, size_t length, void*) {
        _mark_done(request, status, length);
    };

    void* req = ucp_stream_recv_nbx(ep, buf, len, &got, &p);
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error(std::string("ucp_stream_recv_nbx failed: ") + _ucs_str(UCS_PTR_STATUS(req)));
    }
    return req; // nullptr 代表立即完成（且 got>0），否则异步
}

// 组合一个“两段式分帧”的异步写：先发 header 再发 payload；外层 AsyncRequest 聚合两次完成
std::shared_ptr<UcxCommunicator::AsyncRequest>

UcxCommunicator::WriteAsync(const void* buf, size_t size) {
    if (!_ep) throw std::runtime_error("WriteAsync on null endpoint");

    auto areq = std::make_shared<AsyncRequest>();

    // 1) 先发 4B 头
    FrameHeader hdr{ static_cast<uint32_t>(size) };
    void* req_h = _post_stream_send(_ep, &hdr, sizeof(hdr));

    // 2) 再发 payload（可能立刻完成，也可能返回一个异步 req）
    void* req_p = nullptr;
    auto post_payload = [&]() {
        req_p = _post_stream_send(_ep, buf, size);
    };

    // 组合两个完成：当 header 与 payload 都完成时，标记 areq 完成
    auto try_finish = [this, areq, &req_h, &req_p, size]() {
        bool h_done = (req_h == nullptr) ? true : reinterpret_cast<_req_state*>(req_h)->completed;
        bool p_done = (req_p == nullptr) ? true : reinterpret_cast<_req_state*>(req_p)->completed;
        if (h_done && p_done) {
            ucs_status_t st = UCS_OK;
            if (req_h && reinterpret_cast<_req_state*>(req_h)->status != UCS_OK) st = reinterpret_cast<_req_state*>(req_h)->status;
            if (req_p && reinterpret_cast<_req_state*>(req_p)->status != UCS_OK) st = reinterpret_cast<_req_state*>(req_p)->status;
            if (req_h) ucp_request_free(req_h);
            if (req_p) ucp_request_free(req_p);
            areq->status.store(st);
            areq->bytes.store((st == UCS_OK) ? (sizeof(FrameHeader) + (size_t)areq->bytes.load()) : 0);
            areq->completed.store(true);
            if (areq->on_complete) areq->on_complete(st, size);
        }
    };

    // 如果 header 立即完成，直接发 payload
    if (req_h == nullptr) {
        post_payload();
        try_finish();
    } else {
        // 否则由进度线程推进，直到 header 的 req_state->completed=true 时再发 payload
        // 我们在 progress 线程里周期性检查，不额外加锁，简单处理：
        // 为了确保 payload 能够在 header 完成后尽快投递，这里用一个轻量的后台任务：
        std::thread([this, post_payload, try_finish, req_h]() mutable {
            // 等 header 完
            auto* s = reinterpret_cast<_req_state*>(req_h);
            while (!s->completed) {
                if (ucp_worker_progress(_worker) == 0) _wait_progress();
            }
            // 发 payload
            post_payload();
            // 再等二者都完成
            try_finish();
        }).detach();
    }

    return areq;
}

// 组合一个“两段式分帧”的异步读：先收 header，再按长度收 payload
std::shared_ptr<UcxCommunicator::AsyncRequest>
UcxCommunicator::ReadAsync(void* buf, size_t max_size) {
    if (!_ep) throw std::runtime_error("ReadAsync on null endpoint");

    auto areq = std::make_shared<AsyncRequest>();
    auto* hdr = new FrameHeader{}; // 临时头缓存，完成后释放

    // 1) 先收 4B 头
    void* req_h = _post_stream_recv(_ep, hdr, sizeof(FrameHeader));

    auto do_payload = [this, areq, buf, max_size, hdr](void* req_h_in) {
        // 等 header 完
        if (req_h_in != nullptr) {
            auto* s = reinterpret_cast<_req_state*>(req_h_in);
            while (!s->completed) {
                if (ucp_worker_progress(_worker) == 0) _wait_progress();
            }
            if (s->status != UCS_OK) {
                ucs_status_t st = s->status;
                ucp_request_free(req_h_in);
                areq->status.store(st);
                areq->completed.store(true);
                if (areq->on_complete) areq->on_complete(st, 0);
                delete hdr;
                return;
            }
            ucp_request_free(req_h_in);
        }
        // 解析长度并收 payload
        if (hdr->len > max_size) {
            delete hdr;
            areq->status.store(UCS_ERR_MESSAGE_TRUNCATED);
            areq->completed.store(true);
            if (areq->on_complete) areq->on_complete(UCS_ERR_MESSAGE_TRUNCATED, 0);
            return;
        }
        void* req_p = _post_stream_recv(_ep, buf, hdr->len);

        // 等 payload 完
        size_t rx_bytes = 0;
        if (req_p == nullptr) {
            // 立即完成的路径；无法从 immediate path 拿到长度，这里保守设置为 len
            rx_bytes = hdr->len;
        } else {
            auto* sp = reinterpret_cast<_req_state*>(req_p);
            while (!sp->completed) {
                if (ucp_worker_progress(_worker) == 0) _wait_progress();
            }
            rx_bytes = sp->bytes;
            ucs_status_t st = sp->status;
            ucp_request_free(req_p);
            if (st != UCS_OK) {
                delete hdr;
                areq->status.store(st);
                areq->completed.store(true);
                if (areq->on_complete) areq->on_complete(st, 0);
                return;
            }
        }
        areq->bytes.store(rx_bytes);
        areq->status.store(UCS_OK);
        areq->completed.store(true);
        if (areq->on_complete) areq->on_complete(UCS_OK, rx_bytes);
        delete hdr;
    };

    // 如果 header 立即完成则直接处理 payload，否则后台等待 header 完成
    if (req_h == nullptr) {
        do_payload(nullptr);
    } else {
        std::thread([do_payload, req_h]() { do_payload(req_h); }).detach();
    }

    return areq;
}

// ---------------- 阻塞封装（保持原有 API 语义） ----------------
size_t UcxCommunicator::Write(const char* buffer, size_t size) {
    if (!_ep) throw std::runtime_error("Write on null endpoint");
    auto r = WriteAsync(buffer, size);
    // 同步 API：等待完成
    while (!r->completed.load()) {
        if (ucp_worker_progress(_worker) == 0) _wait_progress();
    }
    if (r->status.load() != UCS_OK) {
        throw std::runtime_error(std::string("Write completion: ") + _ucs_str(r->status.load()));
    }
    return size;
}

size_t UcxCommunicator::Read(char* buffer, size_t size) {
    if (!_ep) throw std::runtime_error("Read on null endpoint");
    auto r = ReadAsync(buffer, size);
    // 同步 API：等待完成
    while (!r->completed.load()) {
        if (ucp_worker_progress(_worker) == 0) _wait_progress();
    }
    if (r->status.load() != UCS_OK) {
        throw std::runtime_error(std::string("Read completion: ") + _ucs_str(r->status.load()));
    }
    return r->bytes.load();
}

void UcxCommunicator::Sync() {
    if (!_ep) return;

    ucp_request_param_t p{};
    void* req = ucp_ep_flush_nbx(_ep, &p);
    if (req == nullptr) return;
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
        if (ucp_worker_progress(_worker) == 0) {
            _wait_progress();
        }
    }

    ucp_request_free(req);
}

void UcxCommunicator::Close() {
    if (_closing) return;
    _closing = true;

    StopProgress();

    if (_ep) {
        ucp_request_param_t p{};
        p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
        p.cb.send      = [](void* request, ucs_status_t status, void*) { _mark_done(request, status); };
        void* req = ucp_ep_close_nbx(_ep, &p);
        if (UCS_PTR_IS_PTR(req)) {
            auto* s = reinterpret_cast<_req_state*>(req);
            while (!s->completed) {
                if (ucp_worker_progress(_worker) == 0) _wait_progress();
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
