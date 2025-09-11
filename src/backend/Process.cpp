#include "gvirtus/backend/Process.h"

#include <gvirtus/common/JSON.h>
#include <gvirtus/common/SignalException.h>
#include <gvirtus/common/SignalState.h>
#include "communicators/hybrid/HybridCommunicator.h"
#include <signal.h>
#include <gvirtus/backend/Process.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <functional>
#include <thread>
#include <iostream>

#define DEBUG

using gvirtus::backend::Process;
using gvirtus::common::LD_Lib;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::Endpoint;

using std::chrono::steady_clock;
using namespace std;

Process::Process(std::shared_ptr<LD_Lib<Communicator, std::shared_ptr<Endpoint>>> communicator,
                 vector<string> &plugins)
    : Observable() {
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Process"));

    // Set the logging level
    log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;
    char *val = getenv("GVIRTUS_LOGLEVEL");
    std::string logLevelString = (val == NULL ? std::string("") : std::string(val));
    if (!logLevelString.empty()) {
        logLevel = std::stoi(logLevelString);
    }
    logger.setLogLevel(logLevel);

    signal(SIGCHLD, SIG_IGN);
    _communicator = communicator;
    mPlugins = plugins;
}

// 统一读取“routine 名称”的工具：按 communicator 类型选择读取策略
// 返回 true 表示读取到一条完整的 routine 名；false 表示连接关闭或读取失败
static bool getstring(Communicator *c, std::string &s) {
#ifdef DEBUG
    const char* rtti = "<no-rtti>";
    try { rtti = typeid(*c).name(); } catch (...) {}
    std::string name;
    try { name = c ? c->to_string() : "<null c>";
    } catch (...) { name = "<no to_string()>"; }
    fprintf(stderr, "[getstring] c=%p rtti=%s to_string()=%s\n", (void*)c, rtti, name.c_str());
#endif

    if (!c) return false;

    const std::string comm = c->to_string();

    // 1) 传统 TCP：逐字节直到 '\0'
    if (comm == "tcpcommunicator") {
        s.clear();
        char ch = 0;
        while (c->Read(&ch, 1) == 1) {
            if (ch == 0) return true;
            s.push_back(ch);
        }
        return false;
    }

    // 2) 旧 RDMA 分支：沿用原逻辑（保持兼容）
    if (comm == "rdmacommunicator") {
        try {
            s.clear();
            // 给一个宽裕的缓冲（routine 名通常很短）；RDMA 通道自己有长度边界
            char buf[1024] = {0};
            size_t got = c->Read(buf, sizeof(buf));
            if (got > 0) {
                // 若包含 '\0'，以 '\0' 为终止；否则用 got 的长度
                const char* end = (const char*)memchr(buf, '\0', got);
                if (end) s.assign(buf, end - buf);
                else     s.assign(buf, got);
                return true;
            }
        } catch (...) {
            return false;
        }
        return false;
    }

    // 3) Hybrid / UCX：按“长度分帧”的 Read 一次性取完整 payload，再从中提取 routine
    if (comm == "hybridcommunicator" || comm == "ucxcommunicator") {
        s.clear();
        // UCX 的 Read 是“读满一帧”，如果你给的缓冲太小会抛错；
        // 这里给一个足够大的缓冲区（4KB），远大于 routine 名的长度（一般 < 256）
        char buf[4096];
        memset(buf, 0, sizeof(buf));
        size_t got = 0;
        try {
            got = c->Read(buf, sizeof(buf));
        } catch (const std::exception& e) {
            fprintf(stderr, "[getstring] UCX/Hybrid Read failed: %s\n", e.what());
            return false;
        } catch (...) {
            return false;
        }

        if (got == 0) return false;

        // 如果 payload 以 '\0' 结尾，去掉尾 0；否则按 got 使用
        if (buf[got - 1] == '\0') {
            s.assign(buf, got - 1);
        } else {
            s.assign(buf, got);
        }
        return true;
    }

    // 其他未知 communicator：按 TCP 兼容策略
    {
        s.clear();
        char ch = 0;
        while (c->Read(&ch, 1) == 1) {
            if (ch == 0) return true;
            s.push_back(ch);
        }
        return false;
    }
}

extern std::string getEnvVar(std::string const &key);
std::string getGVirtuSHome() {
    std::string gvirtus_home = getEnvVar("GVIRTUS_HOME");
    return gvirtus_home;
}

void Process::Start() {
    LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] Process::Start() called.");

    for_each(mPlugins.begin(), mPlugins.end(), [this](const std::string &plug) {
        std::string gvirtus_home = getGVirtuSHome();
        std::string to_append = "libgvirtus-plugin-" + plug + ".so";
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] appending " << to_append << ".");
        auto ld_path = fs::path(gvirtus_home + "/lib").append(to_append);
        try {
            auto dl = std::make_shared<LD_Lib<Handler>>(ld_path, "create_t");
            dl->build_obj();
            _handlers.push_back(dl);
        } catch (const std::string &e) {
            LOG4CPLUS_ERROR(logger, e);
        }
    });

    // 每个 client 连接的处理函数
    std::function<void(Communicator *)> execute = [=](Communicator *client_comm) {
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] execute() start");
        std::string routine;
        std::shared_ptr<Buffer> input_buffer = std::make_shared<Buffer>();

        while (getstring(client_comm, routine)) {
            LOG4CPLUS_DEBUG(logger, "✓ - Received routine " << routine);

            // —— Hybrid 分流（保留原有策略）——
            gvirtus::communicators::HybridCommunicator* hybrid = nullptr;
            if (client_comm && client_comm->to_string() == "hybridcommunicator") {
                hybrid = dynamic_cast<gvirtus::communicators::HybridCommunicator*>(client_comm);
            }
            if (hybrid) {
                const bool use_rdma =
                    routine.rfind("cudaRegisterFatBinary", 0) == 0 ||
                    routine.rfind("cudaRegisterFatBinaryEnd", 0) == 0 ||
                    routine.rfind("cudaMemcpyAsync", 0) == 0 ||
                    routine.rfind("cudaMemcpy", 0) == 0;
                if (use_rdma) {
                    hybrid->begin_call(routine, gvirtus::communicators::Transport::RDMA, /*bytes_hint*/1);
                } else {
                    hybrid->begin_call(routine, gvirtus::communicators::Transport::TCP, 0);
                }
            }

            // —— 读取参数缓冲（Buffer 内部会按协议从 client_comm 取 payload）——
            input_buffer->Reset(client_comm);

            // —— 选择可执行的 handler 并执行 —— 
            std::shared_ptr<Handler> h = nullptr;
            for (auto &ptr_el : _handlers) {
                if (ptr_el->obj_ptr()->CanExecute(routine)) {
                    h = ptr_el->obj_ptr();
                    break;
                }
            }

            std::shared_ptr<communicators::Result> result;
            if (h == nullptr) {
                LOG4CPLUS_ERROR(logger, "✖ - [Process " << getpid() << "]: Requested unknown routine " << routine << ".");
                result = std::make_shared<communicators::Result>(-1, std::make_shared<Buffer>());
            } else {
                auto start = steady_clock::now();
                result = h->Execute(routine, input_buffer);
                result->TimeTaken(std::chrono::duration_cast<std::chrono::milliseconds>(
                    steady_clock::now() - start).count() / 1000.0);
            }

            // —— 回写结果：Result::Dump 会根据当前 communicator 把 exit_code / time / outbuf 发回 —— 
            result->Dump(client_comm);

            if (hybrid) {
                hybrid->end_call();
            }

            if (result->GetExitCode() != 0 && routine.compare("cudaLaunch")) {
                LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]: Requested '" << routine << "' routine.");
                LOG4CPLUS_DEBUG(logger, "✓ - - [Process " << getpid() << "]: Exit Code '" << result->GetExitCode() << "'.");
            }
        }

        Notify("process-ended");
    };

    try {
        _communicator->obj_ptr()->Serve();

        while (true) {
            Communicator *client =
                const_cast<Communicator *>(_communicator->obj_ptr()->Accept());
            fprintf(stderr, "[Process] Accept client=%p, comm=%s\n",
                    (void*)client, client ? client->to_string().c_str() : "<null>");

            if (client != nullptr) {
                std::thread(execute, client).detach();
            } else {
                _communicator->obj_ptr()->run();
            }

            if (common::SignalState::get_signal_state(SIGINT)) {
                LOG4CPLUS_DEBUG(logger, "✓ - SIGINT received, killing server on [Process " << getpid() << "]...");
                break;
            }
        }
    } catch (std::string &exc) {
        LOG4CPLUS_ERROR(logger, "✖ - [Process " << getpid() << "]: " << exc);
    }

    LOG4CPLUS_DEBUG(logger, "✓ - Process::Start() returned [Process " << getpid() << "].");
}

Process::~Process() {
    _communicator.reset();
    _handlers.clear();
    mPlugins.clear();
}
