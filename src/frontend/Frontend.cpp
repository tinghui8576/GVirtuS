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
    size_t in_size = input_buffer->GetBufferSize();
    int exit_code = 0;
    double server_exec_sec = 0.0;
    double send_sec = 0.0;
    double recv_sec = 0.0;
    Frontend* frontend = nullptr;
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it == mpFrontends->end()) {
            LOG4CPLUS_ERROR(logger, "Cannot send any job request");
            return;
        }
        frontend = it->second;
    }
    LOG4CPLUS_DEBUG(logger, "DEBUG - Received routine " << routine
                       << " [pid=" << pid << ", tid=" << tid << "]");
    frontend->mRoutinesExecuted++;
    std::string rname(routine);

    if (isSyncRoutine(rname)) {
        // ================== 同步路径 ==================
        auto start_send = steady_clock::now();
        CommandHeader hdr{1};
        frontend->_communicator->obj_ptr()->Write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        frontend->_communicator->obj_ptr()->Write(routine, strlen(routine) + 1);

        if (frontend->_communicator->obj_ptr()->to_string() == "hybridcommunicator") {
            auto* hybrid = dynamic_cast<gvirtus::communicators::HybridCommunicator*>(frontend->_communicator->obj_ptr().get());
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
        frontend->_communicator->obj_ptr()->Read((char *)&exit_code, sizeof(int));
        frontend->mExitCode = exit_code;
        frontend->_communicator->obj_ptr()->Read(reinterpret_cast<char *>(&server_exec_sec), sizeof(server_exec_sec));
        size_t out_buffer_size = 0;
        frontend->_communicator->obj_ptr()->Read((char *)&out_buffer_size, sizeof(size_t));
        
        // CRITICAL FIX 1: Use Reset to correctly parse the Buffer protocol
        if (out_buffer_size > 0) {
            frontend->mpOutputBuffer->Reset(frontend->_communicator->obj_ptr().get());
            frontend->mDataReceived += frontend->mpOutputBuffer->GetBufferSize();
        }

        recv_sec = duration_cast<milliseconds>(steady_clock::now() - start_recv).count() / 1000.0;
        frontend->mRoutineExecutionTime += server_exec_sec;
        frontend->mSendingTime += send_sec;
        frontend->mReceivingTime += recv_sec;
        LOG4CPLUS_DEBUG(logger,
            "Routine '" << routine << "' returned " << exit_code
            << " | server_exec=" << server_exec_sec << "s"
            << " | send=" << send_sec << "s"
            << " | recv=" << recv_sec << "s"
            << " | in=" << in_size << "B"
            << " | out=" << (out_buffer_size > 0 ? frontend->mpOutputBuffer->GetBufferSize() : 0) << "B"
            << " | pid=" << pid << " tid=" << tid);

        if (frontend->_communicator->obj_ptr()->to_string() == "hybridcommunicator") {
            auto hybrid = std::dynamic_pointer_cast<gvirtus::communicators::HybridCommunicator>(frontend->_communicator->obj_ptr());
            if (hybrid) {
                hybrid->end_call();
            }
        }
    } else {
        // ================== 异步路径 ==================
        auto start_send = steady_clock::now();
        CommandHeader hdr{0};
        frontend->_communicator->obj_ptr()->Write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        frontend->_communicator->obj_ptr()->Write(routine, strlen(routine) + 1);
        frontend->mDataSent += in_size;
        input_buffer->Dump(frontend->_communicator->obj_ptr().get());
        send_sec = duration_cast<milliseconds>(steady_clock::now() - start_send).count() / 1000.0;
        frontend->mSendingTime += send_sec;
        LOG4CPLUS_DEBUG(logger,
            "Routine '" << routine << "' launched asynchronously"
            << " | in=" << in_size << "B"
            << " | pid=" << pid << " tid=" << tid);
    }
}

void Frontend::Prepare() {
    pid_t tid = syscall(SYS_gettid);
    {
        if (this->mpFrontends->find(tid) != mpFrontends->end())
            mpFrontends->find(tid)->second->mpInputBuffer->Reset();
    }
}