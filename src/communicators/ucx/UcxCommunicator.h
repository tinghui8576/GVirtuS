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
#include <unordered_map>

// System headers for socket structures
#include <sys/socket.h>
#include <netdb.h>

// UCX API
#include <ucp/api/ucp.h>

// GVirtuS base classes
#include <gvirtus/communicators/Communicator.h>

namespace gvirtus { namespace backend { class Process; } }

namespace gvirtus {
namespace communicators {

class UcxCommunicator : public Communicator {
public:

    // Protocol Magic Number ("GVUX")
    static constexpr uint32_t kMagic        = 0x47565558u;
    // Protocol Version
    static constexpr uint16_t kProtoVersion = 1;

    // Flags for the request header
    enum : uint8_t {
        // Indicates that the client expects a response from the server.
        FLAG_EXPECT_RESPONSE = 1u << 0
    };

    // Structure holding the result of a synchronous SubmitRequest call.
    struct SubmitResult {
        ucs_status_t    transport_status{UCS_OK}; // UCX transport status (UCS_OK if successful)
        int             exit_code{0};             // Backend execution exit code (0 usually means success)
        double          server_exec_sec{0.0};     // Time taken by the server to execute the routine (seconds)
        std::vector<char> out;                    // Payload data returned by the server
    };

    // Ticket for an asynchronous request (currently a placeholder).
    struct AsyncTicket {
        uint64_t msg_id{0};
    };

    // ---- Construction/Destruction ----

    // Constructor behavior is asymmetric based on context:
    // - Client side: Creates the full pipeline (Send/Receive/Progress threads).
    // - Server side: Creates a lightweight endpoint wrapper, does not create its own threads.
    UcxCommunicator(const std::string& hostname, const std::string& port);
    UcxCommunicator(const std::string& hostname, const std::string& port, const std::string& tls);
    UcxCommunicator(ucp_context_h context, ucp_worker_h worker, ucp_ep_h ep);
    ~UcxCommunicator() override;

    // ---- Communicator Base Interface ----
    void   Serve() override;
    void   Connect() override;
    size_t Read(char* buffer, size_t size) override;
    size_t Write(const char* buffer, size_t size) override;
    void   Sync() override;
    void   Close() override;
    const Communicator* const Accept() const override;
    std::string to_string() override;

    // ---- Pipeline Request Interface (External API) ----

    /**
     * Submits a request to the backend.
     *
     * @param routine         The name of the remote function/routine to call.
     * @param payload         Pointer to the data payload.
     * @param payload_len     Size of the payload in bytes.
     * @param expect_response If true, the call blocks until a response is received.
     *                        If false, it returns immediately (fire-and-forget).
     * @return SubmitResult containing the response if expect_response is true.
     */
    SubmitResult SubmitRequest(const char* routine,
                               const void* payload, size_t payload_len,
                               bool expect_response = true);

    /**
     * Submits an asynchronous (fire-and-forget) request.
     * Returns an AsyncTicket immediately.
     */
    AsyncTicket SubmitRequestAsync(const char* routine,
                                   const void* payload, size_t payload_len);
    
    // ---- Thread Management ----

    // The old Start/StopNetwork() are replaced by the more descriptive Start/StopPipeline().
    void StartPipeline();
    void StopPipeline();

    // Progress thread management remains the same.
    void StartProgress();
    void StopProgress();

private:

    // NEW: Grants the backend::Process class access to low-level send/receive functions.
    // This allows the server side to send responses directly and efficiently without
    // going through the send queue.
    friend class gvirtus::backend::Process;

    // Forward declarations for internal structures (defined in .cpp)
    struct _req_state;
    struct Queued;
    struct Waiter;

    // ---- UCX Objects ----
    ucp_context_h  _context  = nullptr;
    ucp_worker_h   _worker   = nullptr;
    ucp_listener_h _listener = nullptr;
    ucp_ep_h       _ep       = nullptr;

    // ---- Internal State ----

    // Flag indicating if this instance is just a wrapper for an accepted server-side endpoint.
    bool _is_server_side_wrapper = false;
    
    // Atomic flag to signal that the communicator is closing down.
    std::atomic<bool> _closing{false};

    // Connection details (Client)
    char        _hostname[256] = {0};
    char        _port[32]      = {0};
    std::string _tls;

    // Listener details (Server)
    struct sockaddr_storage _listen_addr{};
    socklen_t               _listen_addrlen = 0;
    std::string             _bind_ifname;

    // Synchronization for accepting new connections (Server)
    mutable std::mutex              _accept_mtx;
    mutable std::condition_variable _accept_cv;
    mutable std::queue<ucp_ep_h>    _accepted_eps;

    // ---- Progress Thread (Drives UCX background work) ----
    std::atomic<bool> _progress_running{false};
    std::thread       _progress_thread;
    std::mutex        _progress_mtx;
    
    // ========================================================================
    //                         CORE MODIFICATION AREA
    // ========================================================================
    
    // ---- 1. Waiter Management ----

    // Stores all 'Waiters' for synchronous requests awaiting a response.
    // Key: msg_id, Value: Shared pointer to the corresponding Waiter object.
    std::mutex _waiters_mtx;
    std::unordered_map<uint64_t, std::shared_ptr<Waiter>> _waiters;

    // ---- 2. Pipeline Threads (Send/Receive Split) ----

    // Send Thread: Responsible for taking requests from _sendq and transmitting them.
    std::thread                       _send_thread;
    std::mutex                        _send_thread_mtx;
    std::atomic<bool>                 _send_thread_running{false};

    // Receive Thread: Responsible for receiving all responses and waking up
    // the corresponding waiting thread via the _waiters map.
    std::thread                       _recv_thread;
    std::mutex                        _recv_thread_mtx;
    std::atomic<bool>                 _recv_thread_running{false};

    // ---- Send Queue (Producer-Consumer) ----
    std::mutex                        _sendq_mtx;
    std::condition_variable           _sendq_cv;
    std::queue<std::shared_ptr<Queued>> _sendq;
    std::atomic<bool>                 _sendq_stopping{false};

    // ---- Message ID Generator ----
    std::atomic<uint64_t>             _msg_id_gen{0};
    uint64_t _next_msg_id();
    
    // ========================================================================

    // ---- Internal Utilities ----
    void _init_context();
    void _finalize_context();
    void _create_worker();
    void _destroy_worker();
    void _setup_listener();
    void _destroy_listener();
    void _resolve_sockaddr(struct sockaddr_storage& ss, socklen_t& slen) const;

    // Low-level send/receive functions. These are now also directly used by the server (via friend).
    
    // Receives exactly 'nbytes' from the stream.
    void   _recv_stream_exact(ucp_worker_h worker, ucp_ep_h ep, void* buf, size_t nbytes);
    
    // Sends exactly 'nbytes' over the stream.
    void   _send_stream_exact(ucp_ep_h ep, const void* buf, size_t nbytes);

    // ---- 3. Refactored: Thread Loop Functions ----

    // The main loop for the send thread (was _network_loop).
    void _send_loop();

    // The main loop for the receive thread (new).
    void _recv_loop();

    // ---- UCX Callbacks ----
    static void _on_conn_request(ucp_conn_request_h req, void* arg);
    static void _on_ep_error(void* arg, ucp_ep_h ep, ucs_status_t status);
};

} // namespace communicators
} // namespace gvirtus

// ---- Factory Function ----

// Extern "C" factory function to create a UcxCommunicator instance from an Endpoint.
extern "C"
std::shared_ptr<gvirtus::communicators::UcxCommunicator>
create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> end);