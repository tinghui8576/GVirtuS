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
#include <cstdio>   // 确保包含了 printf 和 fflush
#include <typeinfo> // 确保包含了 typeid
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
    
    if (frontend->_communicator->obj_ptr()->to_string() == "ucxcommunicator") {
        // ================== [新 UCX "黑盒子适配器" 逻辑路径] ==================
        auto *ucx = dynamic_cast<gvirtus::communicators::UcxCommunicator*>(
            frontend->_communicator->obj_ptr().get());
        if (!ucx) {
            LOG4CPLUS_FATAL(logger, "FATAL: dynamic_cast to UcxCommunicator failed.");
            return;
        }

        // === [适配器] 步骤 1: 将请求打包成一个格式绝对正确的 "邮包" ===
        Buffer request_packet;
        
        // 1a. 序列化 routine 字符串
        request_packet.AddString(routine);
        
        // 1b. [终极修正] 将 input_buffer 的完整内容，作为一个整体的 "货物"，打包进邮包。
        //     我们先写入它的总长度，再写入它的全部数据。
        size_t input_buffer_total_size = input_buffer->GetBufferSize();
        request_packet.Add(input_buffer_total_size);
        if (input_buffer_total_size > 0) {
            request_packet.Append(input_buffer->GetBuffer(), input_buffer_total_size);
        }
        
        const bool expect_response = isSyncRoutine(rname);

        // === 步骤 2: 通过流水线发送这个打包好的 "邮包" ===
        auto t0 = steady_clock::now();
        auto res = ucx->SubmitRequest("execute_routine", 
                                      request_packet.GetBuffer(), 
                                      request_packet.GetBufferSize(), 
                                      expect_response);
        auto t1 = steady_clock::now();
        
        // === [适配器] 步骤 3: 如果是同步请求，则解开响应 "邮包" ===
        if (expect_response) {
            double wall_time = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
            
            // 3a. 使用返回的 res.out 数据，构造一个只读的响应包 Buffer。
            Buffer response_packet(res.out.data(), res.out.size());
            
            // 3b. 从响应包中，按顺序反序列化出结果。
            int exit_code = response_packet.Get<int>();
            double server_exec_sec = response_packet.Get<double>();

            // 3c. 从响应包中反序列化出最终的输出数据，并填充到前端的 mpOutputBuffer 中。
            frontend->mpOutputBuffer->Reset();
            size_t output_len = response_packet.Get<size_t>();
            if (output_len > 0) {
                char* out_data = response_packet.Get<char>(output_len);
                if (out_data) {
                    frontend->mpOutputBuffer->Append(out_data, output_len);
                    delete[] out_data;
                }
            }
            
            // 更新统计
            frontend->mExitCode = exit_code;
            frontend->mRoutineExecutionTime += server_exec_sec;
            frontend->mDataReceived += output_len;
            frontend->mDataSent += request_packet.GetBufferSize();
            frontend->mSendingTime += wall_time;

            LOG4CPLUS_DEBUG(logger, "SYNC routine '" << routine << "' finished."
                << " | exit_code=" << exit_code
                << " | server_exec=" << server_exec_sec << "s"
                << " | wall_time=" << wall_time << "s"
                << " | in=" << input_buffer->GetBufferSize() << "B, out=" << output_len << "B"
                << " [pid=" << pid << ", tid=" << tid << "]");

        } else {
            double submission_time = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
            frontend->mDataSent += request_packet.GetBufferSize();
            frontend->mSendingTime += submission_time;
        }
        
        return;
    }
    // ================== [旧逻辑路径] ==================
    // ... (旧逻辑保持原样，无需修改)
    printf("[EXECUTE_TRACE] STEP 3.1: Path taken -> OLD LOGIC.\n");
    fflush(stdout);

    auto send_request = [&](uint8_t expect_response) {
        printf("[EXECUTE_TRACE] OLD_LOGIC: Preparing to send request header (expect_response=%d).\n", (int)expect_response);
        fflush(stdout);
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
        printf("[EXECUTE_TRACE] OLD_LOGIC: Request header sent.\n");
        fflush(stdout);
    };

    const bool sync_legacy = isSyncRoutine(rname);
    size_t in_size = input_buffer->GetBufferSize();
    double send_sec = 0.0;
    double recv_sec = 0.0;
    
    printf("[EXECUTE_TRACE] OLD_LOGIC: Is sync routine? %s.\n", sync_legacy ? "Yes" : "No");
    fflush(stdout);

    if (sync_legacy) {
        auto start_send = steady_clock::now();
        send_request(1);
        printf("[EXECUTE_TRACE] OLD_LOGIC_SYNC: Preparing to dump buffer of size %zu.\n", in_size);
        fflush(stdout);
        frontend->mDataSent += in_size;
        input_buffer->Dump(frontend->_communicator->obj_ptr().get());
        printf("[EXECUTE_TRACE] OLD_LOGIC_SYNC: Buffer dumped. Now waiting for response...\n");
        fflush(stdout);
        frontend->mpOutputBuffer->Reset();
        auto start_recv = steady_clock::now();
        int exit_code = 0;
        double server_exec_sec = 0.0;
        size_t out_buffer_size = 0;
        printf("[EXECUTE_TRACE] OLD_LOGIC_SYNC: Reading exit_code...\n"); fflush(stdout);
        frontend->_communicator->obj_ptr()->Read((char *)&exit_code, sizeof(int));
        frontend->mExitCode = exit_code;
        printf("[EXECUTE_TRACE] OLD_LOGIC_SYNC: Reading server_exec_sec...\n"); fflush(stdout);
        frontend->_communicator->obj_ptr()->Read(reinterpret_cast<char *>(&server_exec_sec), sizeof(server_exec_sec));
        printf("[EXECUTE_TRACE] OLD_LOGIC_SYNC: Reading out_buffer_size...\n"); fflush(stdout);
        frontend->_communicator->obj_ptr()->Read((char *)&out_buffer_size, sizeof(size_t));
        frontend->mDataReceived += out_buffer_size;
        if (out_buffer_size > 0) {
            printf("[EXECUTE_TRACE] OLD_LOGIC_SYNC: Reading output buffer of size %zu...\n", out_buffer_size); fflush(stdout);
            frontend->mpOutputBuffer->Reset(frontend->_communicator->obj_ptr().get());
        }
        printf("[EXECUTE_TRACE] OLD_LOGIC_SYNC: All responses received.\n"); fflush(stdout);
        recv_sec = duration_cast<milliseconds>(steady_clock::now() - start_recv).count() / 1000.0;
        frontend->mRoutineExecutionTime += server_exec_sec;
    } else {
        auto start_send = steady_clock::now();
        send_request(0);
        printf("[EXECUTE_TRACE] OLD_LOGIC_ASYNC: Preparing to dump buffer of size %zu.\n", in_size);
        fflush(stdout);
        frontend->mDataSent += in_size;
        input_buffer->Dump(frontend->_communicator->obj_ptr().get());
        printf("[EXECUTE_TRACE] OLD_LOGIC_ASYNC: Buffer dumped. No response expected.\n");
        fflush(stdout);
        send_sec = duration_cast<milliseconds>(steady_clock::now() - start_send).count() / 1000.0;
        frontend->mSendingTime += send_sec;
    }
    
    printf("[EXECUTE_TRACE] STEP 4: Leaving Frontend::Execute.\n");
    fflush(stdout);
}

void Frontend::Prepare() {
    pid_t tid = syscall(SYS_gettid);
    {
        if (this->mpFrontends->find(tid) != mpFrontends->end())
            mpFrontends->find(tid)->second->mpInputBuffer->Reset();
    }
}