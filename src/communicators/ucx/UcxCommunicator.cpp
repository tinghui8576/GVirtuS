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
#include <gvirtus/communicators/UcxProtocol.h>
#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Ucx.h>

using gvirtus::communicators::UcxCommunicator;

// ============================ 工具/常量 ============================
namespace {
inline const char* _ucs_str(ucs_status_t st) { return ucs_status_string(st); }

#define UCS_THROW_IF_NOT_OK(expr, what) do {                                  \
  ucs_status_t __st = (expr);                                                 \
  if (__st != UCS_OK) {                                                       \
    throw std::runtime_error(std::string(what) + ": " + _ucs_str(__st));      \
  }                                                                           \
} while(0)

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
      if (a->sin_addr.s_addr == b->sin_addr.s_addr) { name = p->ifa_name; break; }
    } else if (sa->sa_family == AF_INET6) {
      auto* a = reinterpret_cast<const sockaddr_in6*>(sa);
      auto* b = reinterpret_cast<const sockaddr_in6*>(p->ifa_addr);
      if (memcmp(&a->sin6_addr, &b->sin6_addr, sizeof(in6_addr)) == 0) { name = p->ifa_name; break; }
    }
  }
  freeifaddrs(ifas);
  return name;
}

static std::string route_ifname_to(const struct sockaddr* dst, socklen_t dlen) {
  int s = ::socket(dst->sa_family, SOCK_DGRAM, 0);
  if (s < 0) return "";
  (void)::connect(s, dst, dlen);
  sockaddr_storage local{};
  socklen_t llen = sizeof(local);
  if (getsockname(s, reinterpret_cast<sockaddr*>(&local), &llen) != 0) { close(s); return ""; }
  close(s);
  return ifname_from_sockaddr(reinterpret_cast<sockaddr*>(&local));
}

// 网络字节序工具
template<class T>
T hton_any(T v);
template<> inline uint16_t hton_any<uint16_t>(uint16_t v){ return htons(v); }
template<> inline uint32_t hton_any<uint32_t>(uint32_t v){ return htonl(v); }
template<> inline uint64_t hton_any<uint64_t>(uint64_t v){
  uint64_t hi = htonl(uint32_t(v >> 32));
  uint64_t lo = htonl(uint32_t(v & 0xffffffffu));
  return (lo << 32) | hi;
}
template<class T>
T ntoh_any(T v);
template<> inline uint16_t ntoh_any<uint16_t>(uint16_t v){ return ntohs(v); }
template<> inline uint32_t ntoh_any<uint32_t>(uint32_t v){ return ntohl(v); }
template<> inline uint64_t ntoh_any<uint64_t>(uint64_t v){
  uint64_t hi = ntohl(uint32_t(v >> 32));
  uint64_t lo = ntohl(uint32_t(v & 0xffffffffu));
  return (lo << 32) | hi;
}

} // namespace

// ============================ 协议定义 ============================


// ============================ 内部状态实现 ============================

struct UcxCommunicator::_req_state {
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

struct UcxCommunicator::Waiter {
  std::mutex m;
  std::condition_variable cv;
  bool done{false};
  ucs_status_t transport_status{UCS_OK};
  int exit_code{0};
  double server_exec_sec{0.0};
  std::vector<char> out;
};

struct UcxCommunicator::Queued {
  ReqHdr hdr{};
  std::vector<char> routine; // 包含 '\0'
  std::vector<char> payload;
  bool expect_resp{false};
  std::shared_ptr<Waiter> waiter; // 同步请求才非空
};

// ============================ 收发底层 ============================

void UcxCommunicator::_recv_stream_exact(ucp_worker_h worker, ucp_ep_h ep, void* buf, size_t nbytes) {
  if (nbytes == 0) return;
  size_t total = 0;
  while (total < nbytes) {
    auto req_state = std::make_shared<_req_state>();
    ucp_request_param_t p{};
    p.op_attr_mask   = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    p.user_data      = req_state.get();
    p.cb.recv_stream = [](void* request, ucs_status_t status, size_t length, void* user_data) {
      auto* s = reinterpret_cast<_req_state*>(user_data);
      s->signal(status, length);
      ucp_request_free(request);
    };
    size_t got = 0;
    char* current_buf  = static_cast<char*>(buf) + total;
    size_t current_len = nbytes - total;
    void* req = ucp_stream_recv_nbx(ep, current_buf, current_len, &got, &p);
    if (UCS_PTR_IS_ERR(req)) {
      throw std::runtime_error(std::string("ucp_stream_recv_nbx failed: ") + _ucs_str(UCS_PTR_STATUS(req)));
    }
    if (req == nullptr) { total += got; continue; }
    req_state->wait();
    if (req_state->status.load() != UCS_OK) {
      throw std::runtime_error(std::string("stream recv completion: ") + _ucs_str(req_state->status.load()));
    }
    total += req_state->bytes.load();
  }
}

void UcxCommunicator::_send_stream_exact(ucp_ep_h ep, const void* buf, size_t nbytes) {
  if (nbytes == 0) return;
  auto req_state = std::make_shared<_req_state>();
  ucp_request_param_t p{};
  p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  p.user_data    = req_state.get();
  p.cb.send      = [](void* request, ucs_status_t status, void* user_data) {
    auto* s = reinterpret_cast<_req_state*>(user_data);
    s->signal(status, 0);
    ucp_request_free(request);
  };
  void* req = ucp_stream_send_nbx(ep, buf, nbytes, &p);
  if (UCS_PTR_IS_ERR(req)) {
    throw std::runtime_error(std::string("ucp_stream_send_nbx failed: ") + _ucs_str(UCS_PTR_STATUS(req)));
  }
  if (req == nullptr) return;
  req_state->wait();
  if (req_state->status.load() != UCS_OK) {
    throw std::runtime_error(std::string("stream send completion: ") + _ucs_str(req_state->status.load()));
  }
}

// ============================ 上下文/worker/监听 ============================

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

void UcxCommunicator::_init_context() {
  ucp_params_t params;
  memset(&params, 0, sizeof(params));
  params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_NAME;
  params.features   = UCP_FEATURE_STREAM | UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
  params.name       = "gvirtus_ucx";

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
  if (_context) { ucp_cleanup(_context); _context = nullptr; }
}

void UcxCommunicator::_create_worker() {
  ucp_worker_params_t wparams;
  memset(&wparams, 0, sizeof(wparams));
  wparams.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  wparams.thread_mode = UCS_THREAD_MODE_MULTI;
  UCS_THROW_IF_NOT_OK(ucp_worker_create(_context, &wparams, &_worker), "ucp_worker_create");
}

void UcxCommunicator::_destroy_worker() {
  if (_worker) { ucp_worker_destroy(_worker); _worker = nullptr; }
}

void UcxCommunicator::_setup_listener() {
  ucp_listener_params_t lp;
  memset(&lp, 0, sizeof(lp));
  lp.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  lp.sockaddr.addr    = reinterpret_cast<const struct sockaddr*>(&_listen_addr);
  lp.sockaddr.addrlen = _listen_addrlen;
  lp.conn_handler.cb  = &UcxCommunicator::_on_conn_request;
  lp.conn_handler.arg = this;
  UCS_THROW_IF_NOT_OK(ucp_listener_create(_worker, &lp, &_listener), "ucp_listener_create");
}

void UcxCommunicator::_destroy_listener() {
  if (_listener) { ucp_listener_destroy(_listener); _listener = nullptr; }
}

// ============================ 进度线程 ============================
void UcxCommunicator::StartProgress() {
  std::lock_guard<std::mutex> lk(_progress_mtx);
  if (_progress_running) return;
  _progress_running = true;
  _progress_thread = std::thread([this](){
    while (_progress_running && !_closing) {
      if (_worker) ucp_worker_progress(_worker);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
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

// ============================ 网络线程（核心流水线） ============================

void UcxCommunicator::StartNetwork() {
  std::lock_guard<std::mutex> lk(_net_mtx);
  if (_net_running) return;
  _net_running = true;
  _net_thread = std::thread([this](){ this->_network_loop(); });
}

void UcxCommunicator::StopNetwork() {
  {
    std::lock_guard<std::mutex> lk(_sendq_mtx);
    _sendq_stopping = true;
  }
  _sendq_cv.notify_all();
  {
    std::lock_guard<std::mutex> lk(_net_mtx);
    if (_net_running) {
      _net_running = false;
    }
  }
  if (_net_thread.joinable()) _net_thread.join();
}

void UcxCommunicator::_network_loop() {
  // 串行收发，独占 ep
  while (!_closing) {
    std::shared_ptr<UcxCommunicator::Queued> item;

    // 取队列
    {
      std::unique_lock<std::mutex> lk(_sendq_mtx);
      _sendq_cv.wait(lk, [this]{
        return !_sendq.empty() || _sendq_stopping || _closing;
      });
      if (_closing || _sendq_stopping) break;
      item = _sendq.front();
      _sendq.pop();
    }

    if (!item) continue;

    try {
      // 发送请求头
      ReqHdr wh = item->hdr;
      // 转网络序
      wh.magic       = hton_any<uint32_t>(wh.magic);
      wh.version     = hton_any<uint16_t>(wh.version);
      wh.msg_id      = hton_any<uint64_t>(wh.msg_id);
      wh.routine_len = hton_any<uint32_t>(wh.routine_len);
      wh.payload_len = hton_any<uint32_t>(wh.payload_len);
      _send_stream_exact(_ep, &wh, sizeof(wh));

      // 发送 routine
      if (!item->routine.empty()) {
        _send_stream_exact(_ep, item->routine.data(), item->routine.size());
      }
      // 发送 payload
      if (!item->payload.empty()) {
        _send_stream_exact(_ep, item->payload.data(), item->payload.size());
      }

      if (!item->expect_resp) {
        // 异步：不读取响应，直接下一条
        continue;
      }

      // 读取响应头
      RespHdr rh_n{};
      _recv_stream_exact(_worker, _ep, &rh_n, sizeof(rh_n));

      RespHdr rh{};
      rh.magic          = ntoh_any<uint32_t>(rh_n.magic);
      rh.version        = ntoh_any<uint16_t>(rh_n.version);
      rh.status         = rh_n.status; // ucs_status_t 数值直接透传（16位即可）
      rh.exit_code      = rh_n.exit_code;
      rh.server_exec_sec= rh_n.server_exec_sec; // double 按本机端传，双方相同架构即可；如需跨端，可转字节序处理
      rh.msg_id         = ntoh_any<uint64_t>(rh_n.msg_id);
      rh.out_len        = ntoh_any<uint32_t>(rh_n.out_len);

      if (rh.magic != kMagic || rh.version != kProtoVersion || rh.msg_id != item->hdr.msg_id) {
        throw std::runtime_error("protocol mismatch in response header");
      }

      std::vector<char> out;
      if (rh.out_len > 0) {
        out.resize(rh.out_len);
        _recv_stream_exact(_worker, _ep, out.data(), out.size());
      }

      // 唤醒 waiter
      if (item->waiter) {
        std::lock_guard<std::mutex> lk(item->waiter->m);
        item->waiter->transport_status = static_cast<ucs_status_t>(rh.status);
        item->waiter->exit_code        = rh.exit_code;
        item->waiter->server_exec_sec  = rh.server_exec_sec;
        item->waiter->out.swap(out);
        item->waiter->done = true;
        item->waiter->cv.notify_one();
      }
    } catch (const std::exception& e) {
      if (item->waiter) {
        std::lock_guard<std::mutex> lk(item->waiter->m);
        item->waiter->transport_status = UCS_ERR_IO_ERROR;
        item->waiter->exit_code        = -1;
        item->waiter->server_exec_sec  = 0.0;
        item->waiter->out.clear();
        item->waiter->done = true;
        item->waiter->cv.notify_one();
      }
      // 失败后可选择继续 drain 队列或退出，这里选择退出网络线程
      break;
    }
  }
}

// ============================ 对外 API：生产者入队 ============================

uint64_t UcxCommunicator::_next_msg_id() {
  return ++_msg_id_gen;
}

UcxCommunicator::SubmitResult
UcxCommunicator::SubmitRequest(const char* routine_cstr,
                               const void* payload, size_t payload_len,
                               bool expect_response) {
  if (!_ep) throw std::runtime_error("endpoint not connected");

  const size_t rlen = (routine_cstr ? (strlen(routine_cstr) + 1) : 0);

  auto q = std::make_shared<UcxCommunicator::Queued>();
  q->hdr.magic       = kMagic;
  q->hdr.version     = kProtoVersion;
  q->hdr.flags       = expect_response ? FLAG_EXPECT_RESPONSE : 0;
  q->hdr.reserved    = 0;
  q->hdr.msg_id      = _next_msg_id();
  q->hdr.routine_len = static_cast<uint32_t>(rlen);
  q->hdr.payload_len = static_cast<uint32_t>(payload_len);

  if (rlen) {
    q->routine.resize(rlen);
    memcpy(q->routine.data(), routine_cstr, rlen);
  }
  if (payload_len) {
    q->payload.resize(payload_len);
    memcpy(q->payload.data(), payload, payload_len);
  }

  q->expect_resp = expect_response;

  std::shared_ptr<Waiter> w;
  if (expect_response) {
    w = std::make_shared<Waiter>();
    q->waiter = w;
  }

  // 入队
  {
    std::lock_guard<std::mutex> lk(_sendq_mtx);
    _sendq.push(q);
  }
  _sendq_cv.notify_one();

  SubmitResult res{};
  if (!expect_response) {
    res.transport_status = UCS_OK;
    res.exit_code = 0;
    res.server_exec_sec = 0.0;
    return res;
  }

  // 等待 waiter
  {
    std::unique_lock<std::mutex> lk(w->m);
    w->cv.wait(lk, [&]{ return w->done; });
    res.transport_status = w->transport_status;
    res.exit_code        = w->exit_code;
    res.server_exec_sec  = w->server_exec_sec;
    res.out.swap(w->out);
  }
  return res;
}

UcxCommunicator::AsyncTicket
UcxCommunicator::SubmitRequestAsync(const char* routine_cstr,
                                    const void* payload, size_t payload_len) {
  if (!_ep) throw std::runtime_error("endpoint not connected");

  const size_t rlen = (routine_cstr ? (strlen(routine_cstr) + 1) : 0);

  auto q = std::make_shared<UcxCommunicator::Queued>();
  q->hdr.magic       = kMagic;
  q->hdr.version     = kProtoVersion;
  q->hdr.flags       = 0; // async
  q->hdr.reserved    = 0;
  q->hdr.msg_id      = _next_msg_id();
  q->hdr.routine_len = static_cast<uint32_t>(rlen);
  q->hdr.payload_len = static_cast<uint32_t>(payload_len);

  if (rlen) {
    q->routine.resize(rlen);
    memcpy(q->routine.data(), routine_cstr, rlen);
  }
  if (payload_len) {
    q->payload.resize(payload_len);
    memcpy(q->payload.data(), payload, payload_len);
  }
  q->expect_resp = false;

  {
    std::lock_guard<std::mutex> lk(_sendq_mtx);
    _sendq.push(q);
  }
  _sendq_cv.notify_one();

  AsyncTicket t{};
  t.msg_id = q->hdr.msg_id;
  return t;
}

// ============================ Communicator 接口实现 ============================

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
  // server-side wrapper 同样需要进度 + 网络线程
  StartProgress();
  StartNetwork();
}

UcxCommunicator::~UcxCommunicator() {
  try { Close(); } catch (...) {}
}

void UcxCommunicator::Serve() {
  _resolve_sockaddr(_listen_addr, _listen_addrlen);
  _bind_ifname = ifname_from_sockaddr(reinterpret_cast<const sockaddr*>(&_listen_addr));
  if (_context == nullptr) _init_context();
  if (_worker  == nullptr) _create_worker();
  _setup_listener();
  StartProgress();
  StartNetwork();
}

const gvirtus::communicators::Communicator* const UcxCommunicator::Accept() const {
  ucp_ep_h new_ep = nullptr;
  std::unique_lock<std::mutex> lk(_accept_mtx);
  _accept_cv.wait(lk, [this]{ return !_accepted_eps.empty() || _closing; });
  if (_closing && _accepted_eps.empty()) return nullptr;
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
  StartNetwork();
}

// 下面两个阻塞读写不再对外使用（保持兼容保留），仅供网络线程内部调用
size_t UcxCommunicator::Write(const char* buffer, size_t size) {
  if (!_ep) throw std::runtime_error("Write on null endpoint");
  _send_stream_exact(_ep, buffer, size);
  return size;
}
size_t UcxCommunicator::Read(char* buffer, size_t size) {
  if (!_ep) throw std::runtime_error("Read on null endpoint");
  _recv_stream_exact(_worker, _ep, buffer, size);
  return size;
}

std::string UcxCommunicator::to_string() /*const*/ {
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
  while (ucp_request_check_status(req) == UCS_INPROGRESS) {
    if (_worker) ucp_worker_progress(_worker);
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

  // 停网络线程（先停生产）
  StopNetwork();
  // 停进度
  StopProgress();

  if (_ep) {
    ucp_request_param_t params{};
    params.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    params.flags        = UCP_EP_CLOSE_FLAG_FORCE;
    void* req = ucp_ep_close_nbx(_ep, &params);
    if (req != nullptr && !UCS_PTR_IS_ERR(req)) {
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

// ============================ 回调/监听接受 ============================

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

// ============================ 工厂 ============================

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
