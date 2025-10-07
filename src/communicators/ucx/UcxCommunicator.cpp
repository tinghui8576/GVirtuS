#include "UcxCommunicator.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <ifaddrs.h>
#include <iostream>
#include <netdb.h>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/unistd.h>

#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Ucx.h>
#include <gvirtus/communicators/UcxProtocol.h>

namespace gvirtus::communicators {

// ============================ UCX Toolkit & constant
// ============================
namespace {
// Convert ucx status code to string
inline const char *_ucs_str(ucs_status_t st) { return ucs_status_string(st); }
// ucx excepting throw
#define UCS_THROW_IF_NOT_OK(expr, what)                                        \
  do {                                                                         \
    ucs_status_t __st = (expr);                                                \
    if (__st != UCS_OK) {                                                      \
      throw std::runtime_error(std::string(what) + ": " + _ucs_str(__st));     \
    }                                                                          \
  } while (0)
// netork tool function
static std::string ifname_from_sockaddr(const struct sockaddr *sa) {
  struct ifaddrs *ifas = nullptr;
  if (getifaddrs(&ifas) != 0)
    return "";
  std::string name;
  for (auto *p = ifas; p; p = p->ifa_next) {
    if (!p->ifa_addr)
      continue;
    if (p->ifa_addr->sa_family != sa->sa_family)
      continue;
    if (sa->sa_family == AF_INET) {
      auto *a = reinterpret_cast<const sockaddr_in *>(sa);
      auto *b = reinterpret_cast<const sockaddr_in *>(p->ifa_addr);
      if (a->sin_addr.s_addr == b->sin_addr.s_addr) {
        name = p->ifa_name;
        break;
      }
    } else if (sa->sa_family == AF_INET6) {
      auto *a = reinterpret_cast<const sockaddr_in6 *>(sa);
      auto *b = reinterpret_cast<const sockaddr_in6 *>(p->ifa_addr);
      if (memcmp(&a->sin6_addr, &b->sin6_addr, sizeof(in6_addr)) == 0) {
        name = p->ifa_name;
        break;
      }
    }
  }
  freeifaddrs(ifas);
  return name;
}

static std::string route_ifname_to(const struct sockaddr *dst, socklen_t dlen) {
  int s = ::socket(dst->sa_family, SOCK_DGRAM, 0);
  if (s < 0)
    return "";
  (void)::connect(s, dst, dlen);
  sockaddr_storage local{};
  socklen_t llen = sizeof(local);
  if (getsockname(s, reinterpret_cast<sockaddr *>(&local), &llen) != 0) {
    close(s);
    return "";
  }
  close(s);
  return ifname_from_sockaddr(reinterpret_cast<sockaddr *>(&local));
}
} // namespace

// ============================  The pipeline internal state
// ============================
// Temporarily block a thread until a single UCX asynchronous operation (such as
// a send or receive) completes.
struct UcxCommunicator::_req_state {
  std::atomic<bool> completed{false};
  std::atomic<ucs_status_t> status{UCS_INPROGRESS};
  std::atomic<size_t> bytes{0};
  std::mutex mtx;
  std::condition_variable cv;
  void wait() {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [this] { return completed.load(); });
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
// Block a thread (typically the main thread that calls SubmitRequest) for an
// extended period, until a complete server response matching a specific message
// ID is received.
struct UcxCommunicator::Waiter {
  std::mutex m;
  std::condition_variable cv;
  bool done{false};
  ucs_status_t transport_status{UCS_OK};
  int exit_code{0};
  double server_exec_sec{0.0};
  std::vector<char> out;
};
// The internal queue
struct UcxCommunicator::Queued {
  ReqHdr hdr{};
  std::vector<char> routine;
  std::vector<char> payload;
};

// ============================ Packaging of network I/O functions
// ============================
// Ensure that exactly nbytes bytes are received from the network stream
void UcxCommunicator::_recv_stream_exact(ucp_worker_h worker, ucp_ep_h ep,
                                         void *buf, size_t nbytes) {
  std::cout << "[jero] UcxCommunicator::_recv_stream_exact" << std::endl;
  std::cout << "[jero] ep: " << ep << std::endl;
  std::cout << "[jero] buf: " << buf << std::endl;
  std::cout << "[jero] nbytes: " << nbytes << std::endl;

  if (!ep)
    throw std::runtime_error("Attempt to receive on a null endpoint.");
  if (nbytes == 0)
    return;
  size_t total = 0;
  while (total < nbytes) {
    auto req_state = std::make_shared<_req_state>();
    ucp_request_param_t p{};
    p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    p.user_data = req_state.get();
    p.cb.recv_stream = [](void *request, ucs_status_t status, size_t length,
                          void *user_data) {
      auto *s = reinterpret_cast<_req_state *>(user_data);
      s->signal(status, length);
      ucp_request_free(request);
    };
    size_t got = 0;
    char *current_buf = static_cast<char *>(buf) + total;
    size_t current_len = nbytes - total;
    void *req = ucp_stream_recv_nbx(ep, current_buf, current_len, &got, &p);
    if (UCS_PTR_IS_ERR(req)) {
      throw std::runtime_error(std::string("ucp_stream_recv_nbx failed: ") +
                               _ucs_str(UCS_PTR_STATUS(req)));
    }
    if (req == nullptr) {
      total += got;
      continue;
    }
    req_state->wait();
    if (req_state->status.load() != UCS_OK) {
      throw std::runtime_error(std::string("stream recv completion: ") +
                               _ucs_str(req_state->status.load()));
    }
    total += req_state->bytes.load();
  }

  std::cout << "[jero] UcxCommunicator::_recv_stream_exact done" << std::endl;
}
// Ensure that all nbytes bytes in the buffer are sent completely.
void UcxCommunicator::_send_stream_exact(ucp_ep_h ep, const void *buf,
                                         size_t nbytes) {
  if (!ep)
    throw std::runtime_error("Attempt to send on a null endpoint.");
  if (nbytes == 0)
    return;
  auto req_state = std::make_shared<_req_state>();
  ucp_request_param_t p{};
  p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  p.user_data = req_state.get();
  p.cb.send = [](void *request, ucs_status_t status, void *user_data) {
    auto *s = reinterpret_cast<_req_state *>(user_data);
    s->signal(status, 0);
    ucp_request_free(request);
  };
  void *req = ucp_stream_send_nbx(ep, buf, nbytes, &p);
  if (UCS_PTR_IS_ERR(req)) {
    throw std::runtime_error(std::string("ucp_stream_send_nbx failed: ") +
                             _ucs_str(UCS_PTR_STATUS(req)));
  }
  if (req == nullptr)
    return;
  req_state->wait();
  if (req_state->status.load() != UCS_OK) {
    throw std::runtime_error(std::string("stream send completion: ") +
                             _ucs_str(req_state->status.load()));
  }

  std::cout << "[jero] UcxCommunicator::_send_stream_exact done" << std::endl;
}

// ============================ Initialization and destruction of UCX objects
// ============================
// resolve sockaddr info for ucx
void UcxCommunicator::_resolve_sockaddr(struct sockaddr_storage &ss,
                                        socklen_t &slen) const {
  struct addrinfo hints{};
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *res = nullptr;
  int rc = getaddrinfo(_hostname, _port, &hints, &res);
  if (rc != 0 || !res) {
    std::ostringstream oss;
    oss << "UcxCommunicator: getaddrinfo failed for " << _hostname << ":"
        << _port << " (" << gai_strerror(rc) << ")";
    throw std::runtime_error(oss.str());
  }
  memcpy(&ss, res->ai_addr, res->ai_addrlen);
  slen = static_cast<socklen_t>(res->ai_addrlen);
  freeaddrinfo(res);
}
// initialize ucx global context
void UcxCommunicator::_init_context() {
  ucp_params_t params{};
  params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_NAME;
  params.features = UCP_FEATURE_STREAM;
  params.name = "gvirtus_ucx";
  ucp_config_t *cfg = nullptr;
  UCS_THROW_IF_NOT_OK(ucp_config_read(nullptr, nullptr, &cfg),
                      "ucp_config_read");
  UCS_THROW_IF_NOT_OK(ucp_init(&params, cfg, &_context), "ucp_init");
  ucp_config_release(cfg);
}

void UcxCommunicator::_finalize_context() {
  if (_context) {
    ucp_cleanup(_context);
    _context = nullptr;
  }
}
// create & destroy the worker
void UcxCommunicator::_create_worker() {
  ucp_worker_params_t wparams{};
  wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  wparams.thread_mode = UCS_THREAD_MODE_MULTI;
  UCS_THROW_IF_NOT_OK(ucp_worker_create(_context, &wparams, &_worker),
                      "ucp_worker_create");
}

void UcxCommunicator::_destroy_worker() {
  if (_worker) {
    ucp_worker_destroy(_worker);
    _worker = nullptr;
  }
}

// create & destroy the listener, waiting for connect
void UcxCommunicator::_setup_listener() {
  ucp_listener_params_t lp{};
  lp.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                  UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  lp.sockaddr.addr = reinterpret_cast<const struct sockaddr *>(&_listen_addr);
  lp.sockaddr.addrlen = _listen_addrlen;
  lp.conn_handler.cb = &UcxCommunicator::_on_conn_request;
  lp.conn_handler.arg = this;
  UCS_THROW_IF_NOT_OK(ucp_listener_create(_worker, &lp, &_listener),
                      "ucp_listener_create");
}

void UcxCommunicator::_destroy_listener() {
  if (_listener) {
    ucp_listener_destroy(_listener);
    _listener = nullptr;
  }
}

// ============================ progress thread ============================
void UcxCommunicator::StartProgress() {
  std::lock_guard<std::mutex> lk(_progress_mtx);
  if (_progress_running)
    return;
  _progress_running = true;
  _progress_thread = std::thread([this]() {
    while (_progress_running && !_closing) {
      if (_worker)
        ucp_worker_progress(_worker);
    }
  });
}

void UcxCommunicator::StopProgress() {
  {
    std::lock_guard<std::mutex> lk(_progress_mtx);
    if (!_progress_running)
      return;
    _progress_running = false;
  }
  if (_progress_thread.joinable())
    _progress_thread.join();
}

// ============================ pipeline core thread
// ============================
void UcxCommunicator::StartPipeline() {
  {
    std::lock_guard<std::mutex> lk(_send_thread_mtx);
    if (!_send_thread_running) {
      _send_thread_running = true;
      _send_thread = std::thread([this]() { this->_send_loop(); });
    }
  }
  {
    std::lock_guard<std::mutex> lk(_recv_thread_mtx);
    if (!_recv_thread_running) {
      _recv_thread_running = true;
      _recv_thread = std::thread([this]() { this->_recv_loop(); });
    }
  }
}
void UcxCommunicator::StopPipeline() {
  {
    std::lock_guard<std::mutex> lk(_sendq_mtx);
    _sendq_stopping = true;
  }
  _sendq_cv.notify_all();

  {
    std::lock_guard<std::mutex> lk(_send_thread_mtx);
    if (_send_thread_running) {
      _send_thread_running = false;
    }
  }
  if (_send_thread.joinable())
    _send_thread.join();

  {
    std::lock_guard<std::mutex> lk(_recv_thread_mtx);
    if (_recv_thread_running) {
      _recv_thread_running = false;
    }
  }
  if (_recv_thread.joinable())
    _recv_thread.join();
}

// Logic: Loop continuously -> Retrieve a task from the queue -> Serialize the
// request header ->
//-> Send the header, routine, and payload sequentially -> Loop.
void UcxCommunicator::_send_loop() {
  while (!_closing && _send_thread_running) {
    std::shared_ptr<Queued> item;
    {
      std::unique_lock<std::mutex> lk(_sendq_mtx);
      _sendq_cv.wait(lk, [this] {
        return !_sendq.empty() || _sendq_stopping || _closing;
      });
      if (_closing || _sendq_stopping)
        break;
      item = _sendq.front();
      _sendq.pop();
    }

    if (!item)
      continue;

    try {
      // Create the network byte order header first, and initialize it to zero.
      ReqHdr wh_n{};
      memset(&wh_n, 0, sizeof(ReqHdr));

      // get the header of host byte order
      ReqHdr wh = item->hdr;

      // convert every field
      wh_n.magic = hton_any<uint32_t>(wh.magic);
      wh_n.version = hton_any<uint16_t>(wh.version);
      wh_n.flags = wh.flags;
      wh_n.reserved = wh.reserved;
      wh_n.msg_id = hton_any<uint64_t>(wh.msg_id);
      wh_n.routine_len = hton_any<uint32_t>(wh.routine_len);
      wh_n.payload_len = hton_any<uint32_t>(wh.payload_len);

      _send_stream_exact(_ep, &wh_n, sizeof(wh_n));

      if (!item->routine.empty()) {
        _send_stream_exact(_ep, item->routine.data(), item->routine.size());
      }
      if (!item->payload.empty()) {
        _send_stream_exact(_ep, item->payload.data(), item->payload.size());
      }
    } catch (const std::exception &e) {
      std::cerr << "UcxCommunicator::_send_loop error: " << e.what()
                << std::endl;
      _closing = true;
      break;
    }
  }
}

// Loop -> Block and receive response header -> Deserialize -> Receive payload
// -> Find Waiter object using msg_id ->
//-> Populate the Waiter object with the result -> Wake up the main thread
//(w->cv.notify_one()) -> Loop.
void UcxCommunicator::_recv_loop() {
  while (!_closing && _recv_thread_running) {
    try {
      RespHdr rh_n{};
      _recv_stream_exact(_worker, _ep, &rh_n, sizeof(rh_n));

      RespHdr rh{};
      rh.magic = ntoh_any<uint32_t>(rh_n.magic);
      rh.version = ntoh_any<uint16_t>(rh_n.version);
      rh.status = ntoh_any<uint16_t>(rh_n.status);
      rh.exit_code = ntoh_any<int32_t>(rh_n.exit_code);
      rh.server_exec_sec = rh_n.server_exec_sec;
      rh.msg_id = ntoh_any<uint64_t>(rh_n.msg_id);
      rh.out_len = ntoh_any<uint32_t>(rh_n.out_len);

      if (rh.magic != kMagic || rh.version != kProtoVersion) {
        throw std::runtime_error(
            "Protocol mismatch in received response header.");
      }

      std::vector<char> out;
      if (rh.out_len > 0) {
        size_t payload_len = 0;
        _recv_stream_exact(_worker, _ep, &payload_len, sizeof(size_t));
        if (payload_len > 0) {
          out.resize(payload_len);
          _recv_stream_exact(_worker, _ep, out.data(), payload_len);
        }
      }

      std::shared_ptr<Waiter> w;
      {
        std::lock_guard<std::mutex> lk(_waiters_mtx);
        auto it = _waiters.find(rh.msg_id);
        if (it != _waiters.end()) {
          w = it->second;
        } else {
          std::cerr
              << "Warning: Received response for unknown or timed-out msg_id: "
              << rh.msg_id << std::endl;
          continue;
        }
      }

      if (w) {
        std::lock_guard<std::mutex> lk(w->m);
        w->transport_status = static_cast<ucs_status_t>(rh.status);
        w->exit_code = rh.exit_code;
        w->server_exec_sec = rh.server_exec_sec;
        w->out.swap(out);
        w->done = true;
        w->cv.notify_one();
      }

    } catch (const std::exception &e) {
      std::cerr << "UcxCommunicator::_recv_loop error, closing connection: "
                << e.what() << std::endl;
      _closing = true;

      std::lock_guard<std::mutex> lk(_waiters_mtx);
      for (auto const &[msg_id, waiter_ptr] : _waiters) {
        std::lock_guard<std::mutex> lk_waiter(waiter_ptr->m);
        if (!waiter_ptr->done) {
          waiter_ptr->transport_status = UCS_ERR_CANCELED;
          waiter_ptr->exit_code = -1;
          waiter_ptr->done = true;
          waiter_ptr->cv.notify_one();
        }
      }
      _waiters.clear();

      break;
    }
  }
}

// ============================ API：producer enter the queue
// ============================
// Assign a id for each request.
uint64_t UcxCommunicator::_next_msg_id() { return ++_msg_id_gen; }

UcxCommunicator::SubmitResult
// CORE API
// Submit requests, and it determines whether to return synchronously (blocking)
// or asynchronously based on the `expect_response` parameter. A Producer.
UcxCommunicator::SubmitRequest(const char *routine_cstr, const void *payload,
                               size_t payload_len, bool expect_response) {
  if (!_ep)
    throw std::runtime_error("SubmitRequest: endpoint not connected.");

  const size_t rlen = (routine_cstr ? (strlen(routine_cstr) + 1) : 0);

  auto q = std::make_shared<Queued>();
  q->hdr.magic = kMagic;
  q->hdr.version = kProtoVersion;
  q->hdr.flags = expect_response ? FLAG_EXPECT_RESPONSE : 0;
  q->hdr.reserved = 0;
  q->hdr.msg_id = _next_msg_id();
  q->hdr.routine_len = static_cast<uint32_t>(rlen);
  q->hdr.payload_len = static_cast<uint32_t>(payload_len);

  if (rlen) {
    q->routine.assign(routine_cstr, routine_cstr + rlen);
  }
  if (payload_len) {
    const char *p = static_cast<const char *>(payload);
    q->payload.assign(p, p + payload_len);
  }

  std::shared_ptr<Waiter> w;
  if (expect_response) {
    w = std::make_shared<Waiter>();
    {
      std::lock_guard<std::mutex> lk(_waiters_mtx);
      _waiters[q->hdr.msg_id] = w;
    }
  }

  {
    std::lock_guard<std::mutex> lk(_sendq_mtx);
    _sendq.push(q);
  }
  _sendq_cv.notify_one();

  SubmitResult res{};
  if (!expect_response) {
    res.transport_status = UCS_OK;
    return res;
  }

  {
    std::unique_lock<std::mutex> lk(w->m);
    w->cv.wait(lk, [&] { return w->done; });
    res.transport_status = w->transport_status;
    res.exit_code = w->exit_code;
    res.server_exec_sec = w->server_exec_sec;
    res.out.swap(w->out);
  }

  {
    std::lock_guard<std::mutex> lk(_waiters_mtx);
    _waiters.erase(q->hdr.msg_id);
  }

  return res;
}

UcxCommunicator::AsyncTicket
// The async submit
UcxCommunicator::SubmitRequestAsync(const char *routine_cstr,
                                    const void *payload, size_t payload_len) {
  SubmitRequest(routine_cstr, payload, payload_len, false);
  AsyncTicket t{};
  return t;
}

// ============================ UCX Communicator interface & other standard
// operation ============================
UcxCommunicator::UcxCommunicator(const std::string &hostname,
                                 const std::string &port)
    : _tls("") {
  strncpy(_hostname, hostname.c_str(), sizeof(_hostname) - 1);
  _hostname[sizeof(_hostname) - 1] = '\0';
  strncpy(_port, port.c_str(), sizeof(_port) - 1);
  _port[sizeof(_port) - 1] = '\0';
}

UcxCommunicator::UcxCommunicator(const std::string &hostname,
                                 const std::string &port,
                                 const std::string &tls)
    : _tls(tls) {
  strncpy(_hostname, hostname.c_str(), sizeof(_hostname) - 1);
  _hostname[sizeof(_hostname) - 1] = '\0';
  strncpy(_port, port.c_str(), sizeof(_port) - 1);
  _port[sizeof(_port) - 1] = '\0';
}

UcxCommunicator::UcxCommunicator(ucp_context_h context, ucp_worker_h worker,
                                 ucp_ep_h ep) {
  _context = context;
  _worker = worker;
  _ep = ep;
  _is_server_side_wrapper = true;
}

UcxCommunicator::~UcxCommunicator() {
  try {
    Close();
  } catch (...) {
  }
}

void UcxCommunicator::Serve() {
  _resolve_sockaddr(_listen_addr, _listen_addrlen);
  _bind_ifname =
      ifname_from_sockaddr(reinterpret_cast<const sockaddr *>(&_listen_addr));
  if (_context == nullptr)
    _init_context();
  if (_worker == nullptr)
    _create_worker();
  _setup_listener();
  StartProgress();
}

const Communicator *const UcxCommunicator::Accept() const {
  ucp_ep_h new_ep = nullptr;
  std::unique_lock<std::mutex> lk(_accept_mtx);
  _accept_cv.wait(lk, [this] { return !_accepted_eps.empty() || _closing; });
  if (_closing && _accepted_eps.empty())
    return nullptr;
  new_ep = _accepted_eps.front();
  const_cast<std::queue<ucp_ep_h> &>(_accepted_eps).pop();
  return new UcxCommunicator(_context, _worker, new_ep);
}

void UcxCommunicator::Connect() {
  struct sockaddr_storage ss{};
  socklen_t slen = 0;
  _resolve_sockaddr(ss, slen);
  _bind_ifname = route_ifname_to(reinterpret_cast<const sockaddr *>(&ss), slen);
  if (_context == nullptr)
    _init_context();
  if (_worker == nullptr)
    _create_worker();

  ucp_ep_params_t ep_params{};

  // UCX paramemters (only for testing connect.. not product yet)
  ep_params.field_mask =
      UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
      UCP_EP_PARAM_FIELD_ERR_HANDLER | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;

  // setup the corresponding flgas field
  ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
  ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
  ep_params.err_handler.cb = &UcxCommunicator::_on_ep_error;
  ep_params.err_handler.arg = this;
  ep_params.sockaddr.addr = reinterpret_cast<const sockaddr *>(&ss);
  ep_params.sockaddr.addrlen = slen;

  // pass to ucp create
  UCS_THROW_IF_NOT_OK(ucp_ep_create(_worker, &ep_params, &_ep),
                      "ucp_ep_create");

  // start our pipeline after connecting
  StartProgress();
  StartPipeline();
}

size_t UcxCommunicator::Read(char *buffer, size_t size) {
  if (!_ep)
    throw std::runtime_error("Read on null endpoint");
  _recv_stream_exact(_worker, _ep, buffer, size);
  return size;
}

size_t UcxCommunicator::Write(const char *buffer, size_t size) {
  if (!_ep)
    throw std::runtime_error("Write on null endpoint");
  _send_stream_exact(_ep, buffer, size);
  return size;
}

void UcxCommunicator::Sync() {
  if (!_ep)
    return;
  ucp_request_param_t p{};
  void *req = ucp_ep_flush_nbx(_ep, &p);
  if (req == nullptr)
    return;
  if (UCS_PTR_IS_ERR(req)) {
    throw std::runtime_error(std::string("ucp_ep_flush_nbx failed: ") +
                             _ucs_str(UCS_PTR_STATUS(req)));
  }
  while (ucp_request_check_status(req) == UCS_INPROGRESS) {
    if (_worker)
      ucp_worker_progress(_worker);
  }
  ucs_status_t status = ucp_request_check_status(req);
  ucp_request_free(req);
  if (status != UCS_OK) {
    throw std::runtime_error(std::string("Sync flush completion: ") +
                             _ucs_str(status));
  }
}

void UcxCommunicator::Close() {
  if (_closing.exchange(true))
    return;

  _accept_cv.notify_all();

  if (!_is_server_side_wrapper) {
    StopPipeline();
  }

  StopProgress();

  if (_ep) {
    void *req = ucp_ep_close_nb(_ep, UCP_EP_CLOSE_FLAG_FORCE);
    if (req != nullptr && !UCS_PTR_IS_ERR(req)) {
      while (ucp_request_check_status(req) == UCS_INPROGRESS) {
        if (_worker)
          ucp_worker_progress(_worker);
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

std::string UcxCommunicator::to_string() { return "ucxcommunicator"; }

// ============================ Callback/Listener accepted
// ============================
void UcxCommunicator::_on_conn_request(ucp_conn_request_h req, void *arg) {
  auto *self = reinterpret_cast<UcxCommunicator *>(arg);
  ucp_ep_params_t ep_params{};
  ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
  ep_params.conn_request = req;

  ucp_ep_h ep = nullptr;
  ucs_status_t st = ucp_ep_create(self->_worker, &ep_params, &ep);
  if (st != UCS_OK) {
    std::cerr << "ucp_ep_create on accept failed: " << _ucs_str(st)
              << std::endl;
    return;
  }
  {
    std::lock_guard<std::mutex> lk(self->_accept_mtx);
    self->_accepted_eps.push(ep);
  }
  self->_accept_cv.notify_one();
}

void UcxCommunicator::_on_ep_error(void *arg, ucp_ep_h /*ep*/,
                                   ucs_status_t status) {
  auto *self = reinterpret_cast<UcxCommunicator *>(arg);
  std::cerr << "UCX endpoint error: " << _ucs_str(status)
            << ". Closing communicator." << std::endl;
  self->Close();
}

} // namespace gvirtus::communicators

// ============================ Factory stuffs ============================
extern "C" std::shared_ptr<gvirtus::communicators::UcxCommunicator>
create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> end) {
  auto ep =
      std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Ucx>(end);
  if (!ep) {
    throw std::runtime_error(
        "UCX create_communicator: bad endpoint type (expected Endpoint_Ucx)");
  }
  const std::string hostname = ep->address();
  const std::string port = std::to_string(ep->port());
  return std::make_shared<gvirtus::communicators::UcxCommunicator>(hostname,
                                                                   port);
}