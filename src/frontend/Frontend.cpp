/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 * ... (copyright header remains the same) ...
 */

#include <gvirtus/communicators/CommunicatorFactory.h>
#include <gvirtus/communicators/EndpointFactory.h>
#include <gvirtus/frontend/Frontend.h>
#include <filesystem>
#include "communicators/hybrid/HybridCommunicator.h"
#include "communicators/ucx/UcxCommunicator.h"

#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <chrono>
#include <stdlib.h> /* getenv */

#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using namespace std;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::CommunicatorFactory;
using gvirtus::communicators::EndpointFactory;
using gvirtus::frontend::Frontend;

// NEW: Define the command header to resolve protocol ambiguity
#pragma pack(push, 1)
struct CommandHeader {
    uint8_t expect_response; // 1 for sync routines, 0 for async
};
#pragma pack(pop)


static Frontend msFrontend;
std::mutex gFrontendMutex;
map<pthread_t, Frontend *> *Frontend::mpFrontends = NULL;
static bool initialized = false;

log4cplus::Logger logger;

std::string getEnvVar(std::string const &key) {
    char *env_var = getenv(key.c_str());
    return (env_var == nullptr) ? std::string("") : std::string(env_var);
}

void Frontend::Init(Communicator *c) {
    // This function remains identical to your original version
    log4cplus::BasicConfigurator basicConfigurator;
    basicConfigurator.configure();
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS Frontend"));
    std::string logLevelString = getEnvVar("GVIRTUS_LOGLEVEL");
    log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;
    if (!logLevelString.empty()) {
        try {
            logLevel = static_cast<log4cplus::LogLevel>(std::stoi(logLevelString));
        } catch (const std::exception& e) {
            LOG4CPLUS_ERROR(logger, fs::path(__FILE__).filename() << ":" << __LINE__
                             << ": Exception occurred: " << e.what());
            logLevel = log4cplus::INFO_LOG_LEVEL;
        }
    }
    logger.setLogLevel(logLevel);
    pid_t tid = syscall(SYS_gettid);
    std::string config_path = getEnvVar("GVIRTUS_CONFIG");
    if (config_path.empty()) {
        config_path = getEnvVar("GVIRTUS_HOME") + "/etc/properties.json";
        if (config_path.empty()) {
            config_path = "./properties.json";
        }
    }
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        if (mpFrontends->find(tid) == mpFrontends->end()) {
            Frontend *f = new Frontend();
            mpFrontends->insert(make_pair(tid, f));
        }
    }
    LOG4CPLUS_INFO(logger, "GVirtuS frontend version " + config_path);
    try {
        auto endpoint = EndpointFactory::get_endpoint(config_path);
        mpFrontends->find(tid)->second->_communicator =
            CommunicatorFactory::get_communicator(endpoint);
        mpFrontends->find(tid)->second->_communicator->obj_ptr()->Connect();
    } catch (const std::exception& e) {
        LOG4CPLUS_ERROR(logger, fs::path(__FILE__).filename() << ":" << __LINE__
                         << ":" << " Exception occurred: " << e.what());
        exit(EXIT_FAILURE);
    }
    mpFrontends->find(tid)->second->mpInputBuffer = std::make_shared<Buffer>();
    mpFrontends->find(tid)->second->mpOutputBuffer = std::make_shared<Buffer>();
    mpFrontends->find(tid)->second->mpLaunchBuffer = std::make_shared<Buffer>();
    mpFrontends->find(tid)->second->mExitCode = -1;
    mpFrontends->find(tid)->second->mpInitialized = true;
}

Frontend::~Frontend() {
    // This function remains identical to your original version
    static bool destroying = false;
    if (destroying || mpFrontends == nullptr) return;
    destroying = true;
    std::lock_guard<std::mutex> lock(gFrontendMutex);
    {
        pid_t tid = syscall(SYS_gettid);
        auto env = getenv("GVIRTUS_DUMP_STATS");
        bool dump_stats = env && (strcasecmp(env, "on") == 0 ||
                                  strcasecmp(env, "true") == 0 ||
                                  strcmp(env, "1") == 0);
        for (auto it = mpFrontends->begin(); it != mpFrontends->end();) {
            if (it->second == this) {
                it = mpFrontends->erase(it);
                continue;
            }
            if (dump_stats) {
                std::cerr << "[GVIRTUS_STATS] Executed " << it->second->mRoutinesExecuted
                          << " routine(s) in "
                          << it->second->mRoutineExecutionTime << " second(s)\n"
                          << "[GVIRTUS_STATS] Sent "
                          << it->second->mDataSent / (1024 * 1024.0)
                          << " Mb(s) in "
                          << it->second->mSendingTime << " second(s)\n"
                          << "[GVIRTUS_STATS] Received "
                          << it->second->mDataReceived / (1024 * 1024.0)
                          << " Mb(s) in "
                          << it->second->mReceivingTime << " second(s)\n";
            }
            delete it->second;
            it = mpFrontends->erase(it);
        }
        delete mpFrontends;
        mpFrontends = nullptr;
    }
}

Frontend *Frontend::GetFrontend(Communicator *c) {
    // This function remains identical to your original version
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        if (mpFrontends == nullptr)
            mpFrontends = new map<pthread_t, Frontend *>();
    }
    pid_t tid = syscall(SYS_gettid);
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it != mpFrontends->end())
            return it->second;
    }
    Frontend *f = new Frontend();
    try {
        f->Init(c);
        {
            std::lock_guard<std::mutex> lock(gFrontendMutex);
            mpFrontends->insert(make_pair(tid, f));
        }
    } catch (const std::exception& e) {
        LOG4CPLUS_ERROR(logger, "Error initializing Frontend: " << e.what());
        delete f;
        return nullptr;
    }
    return f;
}

// ---------------- 工具函数：判断是否需要同步 ----------------
// MODIFIED: Added crucial routines that require synchronous behavior
static bool isSyncRoutine(const std::string &routine) {
    return (routine.find("cudaMemcpy") != std::string::npos) ||
           (routine.find("cudaStreamSynchronize") != std::string::npos) ||
           (routine.find("cudaMalloc") != std::string::npos) ||
           (routine.find("cudaFree") != std::string::npos) ||
           (routine.find("cudaRegisterFatBinary") != std::string::npos) ||
           (routine.find("cudaRegisterFunction") != std::string::npos);
}

void Frontend::Execute(const char *routine, const Buffer *input_buffer) {
    if (input_buffer == nullptr) input_buffer = mpInputBuffer.get();

    pid_t tid = syscall(SYS_gettid);
    pid_t pid = getpid();

    Frontend* frontend = nullptr;
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it == mpFrontends->end() || !it->second->mpInitialized) {
            LOG4CPLUS_ERROR(logger, "Frontend not initialized for this thread. Cannot send job request.");
            return;
        }
        frontend = it->second;
    }

    std::string rname(routine);
    LOG4CPLUS_DEBUG(logger, "Executing routine: " << rname << " [pid=" << pid << ", tid=" << tid << "]");
    frontend->mRoutinesExecuted++;

    // ======================================================================
    // UCX 新协议路径 (统一调用接口)
    // ======================================================================
    if (frontend->_communicator->obj_ptr()->to_string() == "ucxcommunicator") {
        auto *ucx = dynamic_cast<gvirtus::communicators::UcxCommunicator*>(
            frontend->_communicator->obj_ptr().get());
        if (!ucx) {
            LOG4CPLUS_FATAL(logger, "FATAL: dynamic_cast to UcxCommunicator failed.");
            // This situation indicates a severe logic error and should likely terminate.
            return;
        }

        const void* payload_ptr = input_buffer->GetBuffer();
        const size_t payload_len = input_buffer->GetBufferSize();
        
        // 1. 根据 routine 名称，决定本次调用是否需要等待响应
        const bool expect_response = isSyncRoutine(rname);

        // 2. 统一调用 SubmitRequest，将同步/异步标志作为参数传递
        auto t0 = steady_clock::now();
        auto res = ucx->SubmitRequest(routine, payload_ptr, payload_len, expect_response);
        auto t1 = steady_clock::now();
        
        // 3. 根据是否为同步调用，进行后续处理和统计
        if (expect_response) {
            // 这是同步调用的处理逻辑
            double wall_time = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;

            // 将返回的结果放入前端的输出缓冲区
            frontend->mpOutputBuffer->Reset();
            if (!res.out.empty()) {
                frontend->mpOutputBuffer->Append(res.out.data(), res.out.size());
                frontend->mDataReceived += res.out.size();
            }

            // 更新统计指标
            frontend->mExitCode = res.exit_code;
            frontend->mRoutineExecutionTime += res.server_exec_sec;
            frontend->mDataSent += payload_len;
            // 对于流水线模型，整个调用的 wall time 是衡量发送+接收开销的更准确指标
            frontend->mSendingTime += wall_time; 
            frontend->mReceivingTime += 0; // 包含在 wall_time 中，单独记为0

            LOG4CPLUS_DEBUG(logger, "SYNC routine '" << routine << "' finished."
                << " | exit_code=" << res.exit_code
                << " | server_exec=" << res.server_exec_sec << "s"
                << " | wall_time=" << wall_time << "s"
                << " | in=" << payload_len << "B, out=" << res.out.size() << "B"
                << " [pid=" << pid << ", tid=" << tid << "]");

        } else {
            // 这是异步调用的处理逻辑 (fire-and-forget)
            double submission_time = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
            
            // 更新统计指标
            frontend->mDataSent += payload_len;
            frontend->mSendingTime += submission_time;

            LOG4CPLUS_DEBUG(logger, "ASYNC routine '" << routine << "' submitted."
                << " | submission_time=" << submission_time << "s"
                << " | in=" << payload_len << "B"
                << " [pid=" << pid << ", tid=" << tid << "]");
        }
        
        return; // UCX 路径处理完毕，函数直接返回
    }

    // ======================================================================
    // 非 UCX: 沿用旧逻辑 (保持原样)
    // ======================================================================
    auto send_request = [&](uint8_t expect_response) {
        uint32_t routine_len = static_cast<uint32_t>(strlen(routine) + 1);
        size_t total_len = sizeof(expect_response) + sizeof(routine_len) + routine_len;
        std::vector<char> buf(total_len);

        char* ptr = buf.data();
        memcpy(ptr, &expect_response, sizeof(expect_response));
        ptr += sizeof(expect_response);
        memcpy(ptr, &routine_len, sizeof(routine_len));
        ptr += sizeof(routine_len);
        memcpy(ptr, routine, routine_len);

        frontend->_communicator->obj_ptr()->Write(buf.data(), buf.size());
    };

    const bool sync_legacy = isSyncRoutine(rname);
    size_t in_size = input_buffer->GetBufferSize();
    double send_sec = 0.0;
    double recv_sec = 0.0;

    if (sync_legacy) {
        // 旧的同步路径
        auto start_send = steady_clock::now();
        send_request(1);

        if (frontend->_communicator->obj_ptr()->to_string() == "hybridcommunicator") {
            auto* hybrid = dynamic_cast<gvirtus::communicators::HybridCommunicator*>(
                frontend->_communicator->obj_ptr().get());
            if (hybrid) {
                if (rname.find("cudaMemcpy") != std::string::npos ||
                    rname.find("cudaRegisterFatBinary") != std::string::npos ||
                    rname.find("cudaRegisterFatBinaryEnd") != std::string::npos ||
                    rname.find("cudaMemcpyAsync") != std::string::npos) {
                    hybrid->begin_call(routine, gvirtus::communicators::Transport::RDMA, in_size);
                } else {
                    hybrid->begin_call(routine, gvirtus::communicators::Transport::TCP, in_size);
                }
            }
        }

        frontend->mDataSent += in_size;
        input_buffer->Dump(frontend->_communicator->obj_ptr().get());
        send_sec = duration_cast<milliseconds>(steady_clock::now() - start_send).count() / 1000.0;
        
        frontend->mpOutputBuffer->Reset();

        auto start_recv = steady_clock::now();
        int exit_code = 0;
        double server_exec_sec = 0.0;
        size_t out_buffer_size = 0;
        frontend->_communicator->obj_ptr()->Read((char *)&exit_code, sizeof(int));
        frontend->mExitCode = exit_code;
        frontend->_communicator->obj_ptr()->Read(reinterpret_cast<char *>(&server_exec_sec), sizeof(server_exec_sec));
        frontend->_communicator->obj_ptr()->Read((char *)&out_buffer_size, sizeof(size_t));

        if (out_buffer_size > 0) {
            frontend->mpOutputBuffer->Reset(frontend->_communicator->obj_ptr().get());
            frontend->mDataReceived += frontend->mpOutputBuffer->GetBufferSize();
        }

        recv_sec = duration_cast<milliseconds>(steady_clock::now() - start_recv).count() / 1000.0;
        
        frontend->mRoutineExecutionTime += server_exec_sec;
        frontend->mSendingTime += send_sec;
        frontend->mReceivingTime += recv_sec;

        LOG4CPLUS_DEBUG(logger, "Routine '" << routine << "' returned " << exit_code
            << " | server_exec=" << server_exec_sec << "s"
            << " | send=" << send_sec << "s"
            << " | recv=" << recv_sec << "s"
            << " | in=" << in_size << "B"
            << " | out=" << (out_buffer_size > 0 ? frontend->mpOutputBuffer->GetBufferSize() : 0) << "B"
            << " [pid=" << pid << ", tid=" << tid << "]");

        if (frontend->_communicator->obj_ptr()->to_string() == "hybridcommunicator") {
            auto hybrid = std::dynamic_pointer_cast<gvirtus::communicators::HybridCommunicator>(
                frontend->_communicator->obj_ptr());
            if (hybrid) {
                hybrid->end_call();
            }
        }
    } else {
        // 旧的异步路径
        auto start_send = steady_clock::now();
        send_request(0);

        frontend->mDataSent += in_size;
        input_buffer->Dump(frontend->_communicator->obj_ptr().get());
        send_sec = duration_cast<milliseconds>(steady_clock::now() - start_send).count() / 1000.0;
        frontend->mSendingTime += send_sec;

        LOG4CPLUS_DEBUG(logger, "Routine '" << routine << "' launched asynchronously"
            << " | in=" << in_size << "B"
            << " [pid=" << pid << ", tid=" << tid << "]");
    }
}

void Frontend::Prepare() {
    pid_t tid = syscall(SYS_gettid);
    {
        if (this->mpFrontends->find(tid) != mpFrontends->end())
            mpFrontends->find(tid)->second->mpInputBuffer->Reset();
    }
}