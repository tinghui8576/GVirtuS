//
// UcxCommunicator.cpp
//

#include "UcxCommunicator.h"

#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <netdb.h>
#include <unistd.h>
#include <sys/poll.h>
#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Ucx.h>

using gvirtus::communicators::UcxCommunicator;

namespace {

// 固定使用的TAG（与帧化模型等价；必要时可扩展为多TAG）
static const ucp_tag_t kDefaultTag = 0xABCDEF001234ULL;

static inline const char* _ucs_str(ucs_status_t st) {
    return ucs_status_string(st);
}

} // namespace

// ---------- helpers: sockaddr 解析 ----------
void UcxCommunicator::_resolve_sockaddr(struct sockaddr_storage& ss, socklen_t& slen) const {
    struct addrinfo hints{};
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags    = 0;

    struct addrinfo* res = nullptr;
    int rc = getaddrinfo(_hostname, _port, &hints, &res);
    if (rc != 0 || !res) {
        std::ostringstream oss;
        oss << "UcxCommunicator: getaddrinfo failed for " << _hostname << ":" << _port
            << " (" << gai_strerror(rc) << ")";
        throw std::runtime_error(oss.str());
    }

    memcpy(&ss, res->ai_addr, res->ai_addrlen);
    slen = static_cast<socklen_t>(res->ai_addrlen);
    freeaddrinfo(res);
}

// ---------- UCX 初始化/清理 ----------
void UcxCommunicator::_init_context() {
    ucp_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask =
        UCP_PARAM_FIELD_FEATURES |
        UCP_PARAM_FIELD_REQUEST_SIZE |
        UCP_PARAM_FIELD_REQUEST_INIT |
        UCP_PARAM_FIELD_NAME;
    params.features      = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP; // Tag + 事件唤醒
    params.request_size  = sizeof(_req_state);
    params.request_init  = [](void* req) {
        auto* s = reinterpret_cast<_req_state*>(req);
        s->completed = false;
        s->status    = UCS_INPROGRESS;
        s->bytes     = 0;
    };
    params.name = "gvirtus_ucx";

    ucp_config_t* cfg = nullptr;
    ucs_status_t st = ucp_config_read(nullptr, nullptr, &cfg);
    if (st != UCS_OK) throw std::runtime_error(std::string("ucp_config_read failed: ") + _ucs_str(st));

    // 可选：根据 _tls 设置 UCX_TLS（若用户传入）
    if (!_tls.empty()) {
        ucp_config_modify(cfg, "TLS", _tls.c_str());
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
    if (st != UCS_OK) throw std::runtime_error(std::string("ucp_worker_create failed: ") + _ucs_str(st));

    if (_use_event_fd) {
        int efd = -1;
        st = ucp_worker_get_efd(_worker, &efd);
        if (st == UCS_OK) {
            _worker_event_fd = efd;
        } else {
            // 不支持事件fd，则退化为纯progress轮询
            _use_event_fd = false;
            _worker_event_fd = -1;
        }
    }
}

void UcxCommunicator::_destroy_worker() {
    if (_worker) {
        ucp_worker_destroy(_worker);
        _worker = nullptr;
        _worker_event_fd = -1;
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
    if (st != UCS_OK) throw std::runtime_error(std::string("ucp_listener_create failed: ") + _ucs_str(st));
}

void UcxCommunicator::_destroy_listener() {
    if (_listener) {
        ucp_listener_destroy(_listener);
        _listener = nullptr;
    }
}

// ---------- 进展/等待 ----------
void UcxCommunicator::_progress() const {
    if (_worker) ucp_worker_progress(_worker);
}

void UcxCommunicator::_wait_progress() const {
    if (!_worker) return;

    if (_use_event_fd && _worker_event_fd >= 0) {
        // arm，如果返回 BUSY，说明已经有事件到达，直接 progress
        ucs_status_t st = ucp_worker_arm(_worker);
        if (st == UCS_ERR_BUSY) {
            ucp_worker_progress(_worker);
            return;
        } else if (st != UCS_OK) {
            // 失败则退化为progress
            ucp_worker_progress(_worker);
            return;
        }

        struct pollfd pfd{};
        pfd.fd = _worker_event_fd;
        pfd.events = POLLIN;

        // 阻塞等待事件
        int rc;
        do {
            rc = ::poll(&pfd, 1, -1);
        } while (rc < 0 && errno == EINTR);

        // 醒来后 progress
        ucp_worker_progress(_worker);
    } else {
        // 退化：忙轮询一小段
        for (int i = 0; i < 1000; ++i) {
            if (ucp_worker_progress(_worker) == 0) {
                // 简单让出CPU
                sched_yield();
            }
        }
    }
}

void UcxCommunicator::_ensure_progress_until(bool (*pred)(void*), void* arg) const {
    while (!pred(arg)) {
        // 先尝试进展，若无进展再进入事件等待
        if (ucp_worker_progress(_worker) == 0) {
            _wait_progress();
        }
    }
}

// ---------- 回调 ----------
void UcxCommunicator::_on_conn_request(ucp_conn_request_h request, void* arg) {
    auto* self = reinterpret_cast<UcxCommunicator*>(arg);

    // 从 request 创建 ep
    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask      = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                                UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                UCP_EP_PARAM_FIELD_USER_DATA;
    ep_params.conn_request    = request;
    ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.user_data       = self; // 可用于 error 回调
    ep_params.err_handler.cb  = &UcxCommunicator::_on_ep_error;
    ep_params.err_handler.arg = self;

    ucp_ep_h new_ep = nullptr;
    ucs_status_t st = ucp_ep_create(self->_worker, &ep_params, &new_ep);
    if (st != UCS_OK) {
        // 无法建立ep，直接丢弃（UCX会清理request）
#ifdef DEBUG
        std::cerr << "ucp_ep_create (from conn_request) failed: " << _ucs_str(st) << std::endl;
#endif
        return;
    }

    // 入队，供 Accept() 取走
    {
        std::lock_guard<std::mutex> lk(self->_accept_mtx);
        self->_accepted_eps.push(new_ep);
    }
    self->_accept_cv.notify_one();
}

void UcxCommunicator::_on_ep_error(void* arg, ucp_ep_h ep, ucs_status_t status) {
#ifdef DEBUG
    std::cerr << "UCX endpoint error: " << _ucs_str(status) << std::endl;
#endif
    // 标志关闭；实际清理由 Close() 统一处理
    auto* self = reinterpret_cast<UcxCommunicator*>(arg);
    (void)self;
    (void)ep;
}

// send 完成
void UcxCommunicator::_on_send_complete(void* request, ucs_status_t status, void* user_data) {
    auto* s = reinterpret_cast<_req_state*>(request);
    s->status = status;
    s->completed = true;
    // NBX: 不自动 free；由上层检查并 ucp_request_free
    (void)user_data;
}

// recv 完成
void UcxCommunicator::_on_recv_complete(void* request, ucs_status_t status,
                                        const ucp_tag_recv_info_t* info, void* user_data) {
    auto* s = reinterpret_cast<_req_state*>(request);
    s->status   = status;
    s->bytes    = (info ? info->length : 0);
    s->completed = true;
    (void)user_data;
}

void UcxCommunicator::_on_ep_close_complete(void* request, ucs_status_t status, void* user_data) {
    auto* s = reinterpret_cast<_req_state*>(request);
    s->status = status;
    s->completed = true;
    (void)user_data;
}

// ---------- 阻塞包装：发送/接收 ----------
size_t UcxCommunicator::_blocking_tag_send(const void* buf, size_t size) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    p.cb.send      = &UcxCommunicator::_on_send_complete;
    p.user_data    = nullptr;

    void* req = ucp_tag_send_nbx(_ep, buf, size, kDefaultTag, &p);
    if (req == nullptr) {
        // 立即完成
        return size;
    }
    if (UCS_PTR_IS_ERR(req)) {
        ucs_status_t st = UCS_PTR_STATUS(req);
        throw std::runtime_error(std::string("ucp_tag_send_nbx failed: ") + _ucs_str(st));
    }

    auto* s = reinterpret_cast<_req_state*>(req);
    auto pred = [](void* arg)->bool {
        auto* s2 = reinterpret_cast<_req_state*>(arg);
        return s2->completed;
    };
    _ensure_progress_until(pred, s);

    ucs_status_t st = ucp_request_check_status(req);
    ucp_request_free(req);
    if (st != UCS_OK) {
        throw std::runtime_error(std::string("send completion status: ") + _ucs_str(st));
    }
    return size;
}

size_t UcxCommunicator::_blocking_tag_recv(void* buf, size_t size) {
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                     UCP_OP_ATTR_FIELD_DATATYPE |
                     UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    p.datatype     = ucp_dt_make_contig(1);
    p.cb.recv      = &UcxCommunicator::_on_recv_complete;

    void* req = ucp_tag_recv_nbx(_worker, buf, size, kDefaultTag, (ucp_tag_t)-1, &p);
    if (req == nullptr) {
        // 立即完成（收到的长度 <= size）
        return size;
    }
    if (UCS_PTR_IS_ERR(req)) {
        ucs_status_t st = UCS_PTR_STATUS(req);
        throw std::runtime_error(std::string("ucp_tag_recv_nbx failed: ") + _ucs_str(st));
    }

    auto* s = reinterpret_cast<_req_state*>(req);
    auto pred = [](void* arg)->bool {
        auto* s2 = reinterpret_cast<_req_state*>(arg);
        return s2->completed;
    };
    _ensure_progress_until(pred, s);

    ucs_status_t st = ucp_request_check_status(req);
    size_t rx = s->bytes;
    ucp_request_free(req);

    if (st != UCS_OK) {
        throw std::runtime_error(std::string("recv completion status: ") + _ucs_str(st));
    }
    // 对齐 RdmaCommunicator 语义：期望收满 size
    return size;
}

// ---------- 构造/析构 ----------
UcxCommunicator::UcxCommunicator(const std::string& hostname, const std::string& port)
: _tls("rc,sm,self")
{
    strncpy(_hostname, hostname.c_str(), sizeof(_hostname)-1);
    strncpy(_port,     port.c_str(),     sizeof(_port)-1);
}

UcxCommunicator::UcxCommunicator(const std::string& hostname, const std::string& port,
                                 const std::string& tls)
: _tls(tls)
{
    strncpy(_hostname, hostname.c_str(), sizeof(_hostname)-1);
    strncpy(_port,     port.c_str(),     sizeof(_port)-1);
}

UcxCommunicator::UcxCommunicator(ucp_context_h context,
                                 ucp_worker_h  worker,
                                 ucp_ep_h      ep) {
    _context = context;
    _worker  = worker;
    _ep      = ep;
    _is_server_side_wrapper = true;

    // 事件fd尝试
    if (_worker) {
        int efd = -1;
        if (ucp_worker_get_efd(_worker, &efd) == UCS_OK) {
            _worker_event_fd = efd;
            _use_event_fd    = true;
        } else {
            _worker_event_fd = -1;
            _use_event_fd    = false;
        }
    }
}

UcxCommunicator::~UcxCommunicator() {
#ifdef DEBUG
    std::cout << "Called ~UcxCommunicator()" << std::endl;
#endif
    try {
        Close();
    } catch (...) {
        // 析构期吞掉异常
    }
    if (!_is_server_side_wrapper) {
        _destroy_listener();
        _destroy_worker();
        _finalize_context();
    }
}

// ---------- 对外接口：Serve/Accept/Connect ----------
void UcxCommunicator::Serve() {
#ifdef DEBUG
    std::cout << "UcxCommunicator::Serve()" << std::endl;
#endif
    if (_context == nullptr) _init_context();
    if (_worker  == nullptr) _create_worker();
    _setup_listener();
}

const gvirtus::communicators::Communicator *const UcxCommunicator::Accept() const {
#ifdef DEBUG
    std::cout << "UcxCommunicator::Accept()" << std::endl;
#endif
    ucp_ep_h new_ep = nullptr;

    // 等待队列中出现新 ep
    {
        std::unique_lock<std::mutex> lk(_accept_mtx);
        _accept_cv.wait(lk, [&]{ return !_accepted_eps.empty(); });
        new_ep = _accepted_eps.front();
        const_cast<std::queue<ucp_ep_h>&>(_accepted_eps).pop();
    }

    // 封装为新的 communicator（共享当前 context/worker）
    return new UcxCommunicator(_context, _worker, new_ep);
}

void UcxCommunicator::Connect() {
#ifdef DEBUG
    std::cout << "UcxCommunicator::Connect()" << std::endl;
#endif
    if (_context == nullptr) _init_context();
    if (_worker  == nullptr) _create_worker();

    struct sockaddr_storage ss{};
    socklen_t slen = 0;
    _resolve_sockaddr(ss, slen);

    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_SOCK_ADDR |
                           UCP_EP_PARAM_FIELD_ERR_HANDLER |
                           UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                           UCP_EP_PARAM_FIELD_USER_DATA;
    ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb  = &UcxCommunicator::_on_ep_error;
    ep_params.err_handler.arg = this;
    ep_params.user_data       = this;

    ep_params.sockaddr.addr    = reinterpret_cast<const struct sockaddr*>(&ss);
    ep_params.sockaddr.addrlen = slen;

    ucs_status_t st = ucp_ep_create(_worker, &ep_params, &_ep);
    if (st != UCS_OK) throw std::runtime_error(std::string("ucp_ep_create failed: ") + _ucs_str(st));

    // 简单握手：执行一次 flush 确保路径建立（可选）
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    void* req = ucp_ep_flush_nbx(_ep, &p);
    if (UCS_PTR_IS_ERR(req)) {
        throw std::runtime_error(std::string("ucp_ep_flush_nbx failed: ") + _ucs_str(UCS_PTR_STATUS(req)));
    } else if (req != nullptr) {
        // 等待完成
        auto* s = reinterpret_cast<_req_state*>(req);
        auto pred = [](void* arg)->bool {
            auto* s2 = reinterpret_cast<_req_state*>(arg);
            return s2->completed;
        };
        _ensure_progress_until(pred, s);
        ucs_status_t st2 = ucp_request_check_status(req);
        ucp_request_free(req);
        if (st2 != UCS_OK) {
            throw std::runtime_error(std::string("ucp_ep_flush completion: ") + _ucs_str(st2));
        }
    }
}

// ---------- Read/Write/Sync/Close ----------
size_t UcxCommunicator::Read(char* buffer, size_t size) {
#ifdef DEBUG
    std::cout << "UcxCommunicator::Read(size=" << size << ")" << std::endl;
#endif
    if (!_ep) throw std::runtime_error("Read on null endpoint");
    // 直接阻塞式接收 size 字节：Tag 语义确保匹配
    return _blocking_tag_recv(buffer, size);
}

size_t UcxCommunicator::Write(const char* buffer, size_t size) {
#ifdef DEBUG
    std::cout << "UcxCommunicator::Write(size=" << size << ")" << std::endl;
#endif
    if (!_ep) throw std::runtime_error("Write on null endpoint");
    return _blocking_tag_send(buffer, size);
}

void UcxCommunicator::Sync() {
#ifdef DEBUG
    std::cout << "UcxCommunicator::Sync()" << std::endl;
#endif
    if (!_ep) return;
    ucp_request_param_t p;
    memset(&p, 0, sizeof(p));
    void* req = ucp_ep_flush_nbx(_ep, &p);
    if (req == nullptr) return;
    if (UCS_PTR_IS_ERR(req)) {
        ucs_status_t st = UCS_PTR_STATUS(req);
        throw std::runtime_error(std::string("ucp_ep_flush_nbx failed: ") + _ucs_str(st));
    }
    auto* s = reinterpret_cast<_req_state*>(req);
    auto pred = [](void* arg)->bool {
        auto* s2 = reinterpret_cast<_req_state*>(arg);
        return s2->completed;
    };
    _ensure_progress_until(pred, s);
    ucs_status_t st = ucp_request_check_status(req);
    ucp_request_free(req);
    if (st != UCS_OK) {
        throw std::runtime_error(std::string("Sync flush completion: ") + _ucs_str(st));
    }
}

void UcxCommunicator::Close() {
#ifdef DEBUG
    std::cout << "UcxCommunicator::Close()" << std::endl;
#endif
    if (_closing) return;
    _closing = true;

    // 关闭 endpoint
    if (_ep) {
        ucp_request_param_t p;
        memset(&p, 0, sizeof(p));
        p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        p.cb.send      = &UcxCommunicator::_on_ep_close_complete;

        void* req = ucp_ep_close_nbx(_ep, &p);
        if (UCS_PTR_IS_PTR(req)) {
            auto* s = reinterpret_cast<_req_state*>(req);
            auto pred = [](void* arg)->bool {
                auto* s2 = reinterpret_cast<_req_state*>(arg);
                return s2->completed;
            };
            _ensure_progress_until(pred, s);
            ucp_request_free(req);
        } // NULL => 立即完成；ERR => 仅记录并继续清理
        _ep = nullptr;
    }

    // server 侧的 listener / worker / context 不在 server-side-wrapper 模式中销毁
    if (!_is_server_side_wrapper) {
        _destroy_listener();
        _destroy_worker();
        _finalize_context();
    }
}

extern "C" std::shared_ptr<gvirtus::communicators::UcxCommunicator>
create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    auto ep = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
    if (!ep) {
        throw std::runtime_error("UCX create_communicator: bad endpoint type (expected Endpoint_Ucx)");
    }

    const std::string hostname = ep->address();
    const std::string port     = std::to_string(ep->port());

    // 直接使用已存在的 (host, port) 构造函数
    return std::make_shared<gvirtus::communicators::UcxCommunicator>(hostname, port);
}
