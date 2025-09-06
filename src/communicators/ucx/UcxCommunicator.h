//
// Created by <your name> on <date>.
//
#ifndef GVIRTUS_UCXCOMMUNICATOR_H
#define GVIRTUS_UCXCOMMUNICATOR_H

#include "gvirtus/communicators/Communicator.h"

#include <ucp/api/ucp.h>
#include <ucs/type/status.h>
#include <netdb.h>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>

#define UCX_BACKLOG 8
#define UCX_SMALLBUF_SIZE (1024 * 5)
// #define DEBUG  // enable if needed

/**
 * @brief UcxCommunicator represents a communication interface using UCX (Unified Communication X).
 *
 * 设计要点：
 * - 与 RdmaCommunicator 的对外接口保持一致（Serve/Accept/Connect/Read/Write/Close/Sync）。
 * - 内部采用 UCP 层：listener/ep、非阻塞 send/recv、worker progress 驱动完成。
 * - 提供事件 fd（worker efd）用于低占用 wait，与 progress 结合。
 * - 默认使用 Tag 语义（可在实现中切换为 Active Message）。
 */
namespace gvirtus::communicators {
    class UcxCommunicator : public Communicator {
    public:
        UcxCommunicator() = default;
        UcxCommunicator(const std::string& hostname, const std::string& port);
        UcxCommunicator(const std::string& hostname, const std::string& port,
                        const std::string& tls); // 例如 "rc,sm,self" 或包含 tcp
        // 服务端在 Accept 时用到的构造：由已建立的 endpoint 封装
        explicit UcxCommunicator(ucp_context_h context,
                                 ucp_worker_h  worker,
                                 ucp_ep_h      ep);

        ~UcxCommunicator() override;

        /** Server-side: bind & listen */
        void Serve();
        /** Server-side: accept a new connection and return a new Communicator */
        const Communicator *const Accept() const;

        /** Client-side: connect to server */
        void Connect();

        /** Blocking semantics kept: size bytes read/written before返回 */
        size_t Read(char * buffer, size_t size);
        size_t Write(const char * buffer, size_t size);

        /** Flush/ensure completion as needed（实现中用 progress + 完成计数） */
        void Sync();

        /** Close endpoint & cleanup */
        void Close();

        std::string to_string() { return "ucxcommunicator"; }

    private:
        // ---- UCX core handles ----
        ucp_context_h _context {nullptr};
        ucp_worker_h  _worker  {nullptr};
        ucp_listener_h _listener {nullptr};   // server-side
        ucp_ep_h      _ep      {nullptr};     // connected endpoint

        // ---- addressing / config ----
        char _hostname[256] {0};
        char _port[6] {0};
        std::string _tls;   // UCX_TLS 选择（如 rc,sm,self,tcp）
        bool _is_server_side_wrapper {false}; // 通过 Accept() 构造的实例

        // ---- worker wait (event fd) support ----
        int _worker_event_fd {-1};
        bool _use_event_fd {true};

        // ---- small-buffer staging for compatibility（可让 UCX 直接处理）----
        // 注：UCX 自带 MR 缓存与调度；此缓冲仅在需要时用于兼容上层
        char _smallbuf[UCX_SMALLBUF_SIZE] {0};

        // ---- server accept queue ----
        mutable std::mutex _accept_mtx;
        mutable std::condition_variable _accept_cv;
        mutable std::queue<ucp_ep_h> _accepted_eps;

        // ---- lifecycle flags ----
        bool _closing {false};

        // ---- helpers (defined in .cpp) ----
        void _init_context();
        void _finalize_context();
        void _create_worker();
        void _destroy_worker();
        void _setup_listener();   // server bind
        void _destroy_listener();

        // 统一的进展/等待：在实现中结合 event-fd + ucp_worker_progress
        void _progress() const;
        void _wait_progress() const;     // 事件驱动等待
        void _ensure_progress_until(bool (*pred)(void*), void* arg) const;

        // 发送/接收（阻塞包装在 .cpp 中用非阻塞 + 完成回调实现）
        size_t _blocking_tag_send(const void* buf, size_t size);
        size_t _blocking_tag_recv(void* buf, size_t size);

        // ---- callbacks (static) ----
        // connection request on server listener
        static void _on_conn_request(ucp_conn_request_h request, void* arg);
        // endpoint error handler (disconnect/reset/timeout)
        static void _on_ep_error(void* arg, ucp_ep_h ep, ucs_status_t status);

        // send completion (NBX)
        static void _on_send_complete(void* request, ucs_status_t status, void* user_data);
        // recv completion (NBX)
        static void _on_recv_complete(void* request, ucs_status_t status,
                                      const ucp_tag_recv_info_t* info, void* user_data);

        // endpoint close completion
        static void _on_ep_close_complete(void* request, ucs_status_t status, void* user_data);

        // ---- tiny request state for blocking包装 ----
        struct _req_state {
            volatile bool completed {false};
            ucs_status_t status {UCS_INPROGRESS};
            size_t bytes {0};
            // 可扩展：指向外部计数器/目标大小等
        };

        // 工具：将 sockaddr 解析出来（client/server 通用）
        void _resolve_sockaddr(struct sockaddr_storage& ss, socklen_t& slen) const;

        // 禁用拷贝，仅允许移动（如需）
        UcxCommunicator(const UcxCommunicator&) = delete;
        UcxCommunicator& operator=(const UcxCommunicator&) = delete;
    };
}

#endif // GVIRTUS_UCXCOMMUNICATOR_H
