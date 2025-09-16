#include "gvirtus/backend/Process.h"

#include <gvirtus/common/JSON.h>
#include <gvirtus/common/SignalException.h>
#include <gvirtus/common/SignalState.h>
#include "communicators/hybrid/HybridCommunicator.h"
#include "communicators/ucx/UcxCommunicator.h"
#include <signal.h>
#include <gvirtus/backend/Process.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <functional>
#include <thread>
#include <iostream>
#include <vector>

// 协议头（常量、结构）从公共头取
#include <gvirtus/communicators/UcxProtocol.h>

#define DEBUG
using namespace gvirtus::communicators;
using gvirtus::backend::Process;
using gvirtus::common::LD_Lib;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::Endpoint;

using std::chrono::steady_clock;
using namespace std;

// 网络字节序辅助（只在本文件用）
namespace {
    inline uint64_t ntohll(uint64_t v) {
        uint32_t hi = ntohl(static_cast<uint32_t>(v >> 32));
        uint32_t lo = ntohl(static_cast<uint32_t>(v & 0xffffffffULL));
        return (static_cast<uint64_t>(lo) << 32) | hi;
    }
}

// NEW: 与旧代码兼容的命令头（保留，但当前 UCX 流水线不再使用它）
#pragma pack(push, 1)
struct CommandHeader {
    uint8_t expect_response;
};
#pragma pack(pop)

Process::Process(std::shared_ptr<LD_Lib<Communicator, std::shared_ptr<Endpoint>>> communicator, vector <string> &plugins) : Observable() {
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Process"));
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

// 兼容各 communicator 的 getstring（保留）
bool getstring(Communicator *c, string &s) {
    s.clear();
    if (!c) return false;
    const std::string comm_type = c->to_string();

    if (comm_type == "tcpcommunicator" ||
        comm_type == "hybridcommunicator" ||
        comm_type == "ucxcommunicator") {
        char ch = 0;
        try {
            while (c->Read(&ch, 1) == 1) {
                if (ch == '\0') return true;
                s.push_back(ch);
            }
            return false;
        } catch (const std::exception& e) {
            LOG4CPLUS_WARN(log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("getstring")), "Read failed: " << e.what());
            return false;
        }
    }
    else if (comm_type == "rdmacommunicator") {
        try {
            std::vector<char> buf(1024, 0);
            size_t bytes_read = c->Read(buf.data(), buf.size() - 1);
            if (bytes_read > 0) {
                s.assign(buf.data());
                size_t first_null = s.find('\0');
                if (first_null != std::string::npos) s.resize(first_null);
                return true;
            }
        } catch (const std::exception& e) {
             LOG4CPLUS_ERROR(log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("getstring")), "RDMA Exception: " << e.what());
        }
        return false;
    }
    LOG4CPLUS_ERROR(log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("getstring")), "Unknown communicator type: " << comm_type);
    return false;
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
        if (gvirtus_home.empty()) {
            LOG4CPLUS_FATAL(logger, "GVIRTUS_HOME environment variable is not set.");
            throw std::runtime_error("GVIRTUS_HOME not set");
        }
        std::string to_append = "libgvirtus-plugin-" + plug + ".so";
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] appending " << to_append << ".");
        auto ld_path = fs::path(gvirtus_home).append("lib").append(to_append);
        try {
            auto dl = std::make_shared<LD_Lib<Handler>>(ld_path.string(), "create_t");
            dl->build_obj();
            _handlers.push_back(dl);
        }
        catch (const std::string &e) {
            LOG4CPLUS_ERROR(logger, e);
        }
    });

    std::function<void(Communicator *)> execute = [this](Communicator *client_comm) {
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] New client thread started.");
        std::shared_ptr<Buffer> input_buffer = std::make_shared<Buffer>();

        auto* ucx = dynamic_cast<gvirtus::communicators::UcxCommunicator*>(client_comm);
        if (!ucx) {
            LOG4CPLUS_ERROR(logger, "✖ - Backend process only supports UcxCommunicator.");
            delete client_comm;
            return;
        }

        while (true) {
            try {
                ReqHdr hdr_n{};
                ucx->Read(reinterpret_cast<char*>(&hdr_n), sizeof(hdr_n));

                ReqHdr hdr{};
                hdr.magic       = ntoh_any<uint32_t>(hdr_n.magic);
                hdr.version     = ntoh_any<uint16_t>(hdr_n.version);
                hdr.flags       = hdr_n.flags;
                hdr.msg_id      = ntoh_any<uint64_t>(hdr_n.msg_id);
                hdr.routine_len = ntoh_any<uint32_t>(hdr_n.routine_len);
                hdr.payload_len = ntoh_any<uint32_t>(hdr_n.payload_len);

                if (hdr.magic != kMagic || hdr.version != kProtoVersion) {
                    LOG4CPLUS_ERROR(logger, "Protocol mismatch, closing session.");
                    break;
                }

                std::string routine;
                if (hdr.routine_len > 0) {
                    std::vector<char> rbuf(hdr.routine_len);
                    ucx->Read(rbuf.data(), rbuf.size());
                    routine.assign(rbuf.data()); 
                }

                input_buffer->Reset();
                if (hdr.payload_len > 0) {
                    std::vector<char> payload_vec(hdr.payload_len);
                    ucx->Read(payload_vec.data(), payload_vec.size());
                    input_buffer->Append(payload_vec.data(), payload_vec.size());
                }

                LOG4CPLUS_DEBUG(logger, "✓ - Received routine '" << routine << "' [msg_id=" << hdr.msg_id << "]");

                std::shared_ptr<Handler> h = nullptr;
                for (auto &ptr_el : _handlers) {
                    if (ptr_el->obj_ptr()->CanExecute(routine)) {
                        h = ptr_el->obj_ptr();
                        break;
                    }
                }

                std::shared_ptr<communicators::Result> result;
                if (!h) {
                    LOG4CPLUS_ERROR(logger, "✖ - Unknown routine: " << routine);
                    result = std::make_shared<communicators::Result>(-1, std::make_shared<Buffer>());
                } else {
                    auto start = steady_clock::now();
                    result = h->Execute(routine, input_buffer);
                    result->TimeTaken(std::chrono::duration_cast<std::chrono::milliseconds>(
                        steady_clock::now() - start).count() / 1000.0);
                }

                if (hdr.flags & FLAG_EXPECT_RESPONSE) {
                    // [最终修正]
                    // 1. 使用我们添加的 getter 来安全地获取输出缓冲区
                    auto out_buffer = result->GetOutputBuffer();
                    
                    RespHdr rh{};
                    rh.magic          = kMagic;
                    rh.version        = kProtoVersion;
                    rh.status         = (result->GetExitCode() == 0) ? 0 : 1;
                    rh.exit_code      = result->GetExitCode();
                    rh.server_exec_sec= result->TimeTaken();
                    rh.msg_id         = hdr.msg_id;
                    // 2. 根据缓冲区是否存在且有内容，来设置标志位
                    rh.out_len        = (out_buffer && out_buffer->GetBufferSize() > 0) ? 1 : 0;

                    RespHdr rh_n{};
                    rh_n.magic          = hton_any<uint32_t>(rh.magic);
                    rh_n.version        = hton_any<uint16_t>(rh.version);
                    rh_n.status         = hton_any<uint16_t>(rh.status);
                    rh_n.exit_code      = hton_any<int32_t>(rh.exit_code);
                    rh_n.server_exec_sec= rh.server_exec_sec;
                    rh_n.msg_id         = hton_any<uint64_t>(rh.msg_id);
                    rh_n.out_len        = hton_any<uint32_t>(rh.out_len);
                    
                    ucx->Write(reinterpret_cast<const char*>(&rh_n), sizeof(rh_n));

                    // 3. 如果标志位被设置，则调用 Result 已有的 Dump 方法来发送数据
                    if (rh.out_len > 0) {
                        result->Dump(ucx);
                    }
                    
                    LOG4CPLUS_DEBUG(logger, "✓ - Sent response for msg_id=" << hdr.msg_id);
                }
                
            } catch (const std::exception& e) {
                LOG4CPLUS_WARN(logger, "Exception in client thread, closing session: " << e.what());
                break;
            }
        }
        delete client_comm;
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] Client thread finished.");
        Notify("process-ended");
    };

    try {
        _communicator->obj_ptr()->Serve();
        while (true) {
            if (common::SignalState::get_signal_state(SIGINT)) {
                LOG4CPLUS_DEBUG(logger, "✓ - SIGINT received, shutting down server...");
                _communicator->obj_ptr()->Close();
                break;
            }
            
            Communicator *client = const_cast<Communicator *>(_communicator->obj_ptr()->Accept());
            
            if (client != nullptr) {
                std::thread(execute, client).detach();
            } else {
                LOG4CPLUS_INFO(logger, "Accept returned null, server is likely shutting down.");
                break; 
            }
        }
    }
    catch (const std::exception &exc) {
        LOG4CPLUS_ERROR(logger, "✖ - Exception in main server loop: " << exc.what());
    }

    LOG4CPLUS_DEBUG(logger, "✓ - Process::Start() returned.");
}

Process::~Process() {
    _communicator.reset();
    _handlers.clear();
    mPlugins.clear();
}
