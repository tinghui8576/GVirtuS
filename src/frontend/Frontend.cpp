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
#include <cstdio>   
#include <typeinfo> 
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <vector>
#include <cstring>
#include <chrono>
#include <stdlib.h> 
#include <sstream>
#include <iomanip>
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

// ---------------- Utility function: Determine if synchronization is needed ----------------
// MODIFIED: Added crucial routines that require synchronous behavior
static bool isSyncRoutine(const std::string &routine) {
    return (routine.find("cudaMemcpy") != std::string::npos) ||
           (routine.find("cudaStreamSynchronize") != std::string::npos) ||
           (routine.find("cudaMalloc") != std::string::npos) ||
           (routine.find("cudaFree") != std::string::npos) ||
           (routine.find("cudaPopCallConfiguration") != std::string::npos) ||
           (routine.find("cudaPopCall") != std::string::npos) ||
           (routine.find("cudaRegisterFatBinary") != std::string::npos) ||
           (routine.find("cudaRegisterFunction") != std::string::npos);
}

void Frontend::Execute(const char *routine, const Buffer *input_buffer) {
    printf("[FE] RUN: Now execute runing.\n");
    fflush(stdout);
    std::cout << "[FE] COUT: Now executing Frontend::Execute()" << std::endl;
    if (input_buffer == nullptr) input_buffer = mpInputBuffer.get();

    pid_t tid = syscall(SYS_gettid);
    pid_t pid = getpid();

    Frontend* frontend = nullptr;
    {
        std::lock_guard<std::mutex> lock(gFrontendMutex);
        auto it = mpFrontends->find(tid);
        if (it == mpFrontends->end() || !it->second->mpInitialized) {
            printf("[FE] ERROR: Frontend not initialized for this thread. Cannot send job request.\n");
            fflush(stdout);
            return;
        }
        frontend = it->second;
    }

    std::string rname = routine ? std::string(routine) : std::string("");
    printf("[FE] Execute begin: routine=\"%s\" pid=%d tid=%d\n", rname.c_str(), (int)pid, (int)tid);
    fflush(stdout);

    frontend->mRoutinesExecuted++;

    auto print_hex_prefix = [](const char* tag, const void* p, size_t n, size_t limit = 32) {
        const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
        size_t m = (n < limit ? n : limit);
        printf("%s [len=%zu] hex:", tag, n);
        for (size_t i = 0; i < m; ++i) printf(" %02X", (unsigned int)b[i]);
        if (n > limit) printf(" ...");
        printf("\n");
        fflush(stdout);
    };

    if (frontend->_communicator->obj_ptr()->to_string() == std::string("ucxcommunicator")) {
        auto *ucx = dynamic_cast<gvirtus::communicators::UcxCommunicator*>(
            frontend->_communicator->obj_ptr().get());
        if (!ucx) {
            printf("[FE] FATAL: dynamic_cast<UcxCommunicator> failed.\n");
            fflush(stdout);
            return;
        }

        // ------- Step 0: 参数区准备（必要时剥掉重复的 "routine\\0" 前缀） -------
        const char* in_buf = input_buffer->GetBuffer();
        size_t in_len = input_buffer->GetBufferSize();
        print_hex_prefix("[FE] input_buffer head", in_buf, in_len, 32);

        size_t param_offset = 0;
        if (!rname.empty() && in_len >= rname.size() + 1) {
            bool exact_dup = (::memcmp(in_buf, rname.c_str(), rname.size()) == 0) &&
                             (in_buf[rname.size()] == '\0');
            if (exact_dup) {
                param_offset = rname.size() + 1;
                printf("[FE] WARN: Detected duplicated routine prefix in input_buffer, trimming %zu bytes (\"%s\\0\").\n",
                       param_offset, rname.c_str());
                fflush(stdout);
            }
        }
        const char* param_ptr = in_buf + param_offset;
        size_t      param_len = (in_len >= param_offset) ? (in_len - param_offset) : 0;

        // ------- Step 1: 构造请求 payload： [string] [size_t] [bytes] -------
        Buffer request_packet;
        request_packet.AddString(rname.c_str());   // 双长度头
        request_packet.Add(param_len);
        if (param_len > 0) {
            request_packet.Append(param_ptr, param_len);
        }

        // 期望长度：两个 size_t + 字符串本体 + 参数长度 + 参数数据
        const size_t str_len = rname.size() + 1; // 含 '\0'
        const size_t expected_payload_len =
            2*sizeof(size_t) + str_len + sizeof(size_t) + param_len;

        if (request_packet.GetBufferSize() != expected_payload_len) {
            printf("[FE][UCX] ERROR: Payload length mismatch (double-len). built=%zu expected=%zu\n",
                request_packet.GetBufferSize(), expected_payload_len);
            print_hex_prefix("[FE][UCX] payload head", request_packet.GetBuffer(),
                            request_packet.GetBufferSize(), 64);
            return; // 协议不一致就别发，免得对端更乱
        }

        printf("[FE][UCX] Payload ready. routine=\"%s\" param_len=%zu payload_size=%zu\n",
               rname.c_str(), param_len, request_packet.GetBufferSize());
        print_hex_prefix("[FE][UCX] payload head", request_packet.GetBuffer(),
                         request_packet.GetBufferSize(), 64);

        // ------- Step 1.5: 本地“预反解”验证 -------
            
        {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(request_packet.GetBuffer());
        size_t remain = request_packet.GetBufferSize();

        auto read_size_t = [&](size_t &out) -> bool {
            if (remain < sizeof(size_t)) return false;
            ::memcpy(&out, p, sizeof(size_t));
            p += sizeof(size_t); remain -= sizeof(size_t);
            return true;
        };
        auto read_bytes = [&](size_t n, const unsigned char* &out) -> bool {
            if (remain < n) return false;
            out = p; p += n; remain -= n;
            return true;
        };

        size_t len1=0, len2=0;
        if (!read_size_t(len1) || !read_size_t(len2)) {
            printf("[FE][UCX][PreParse] ERROR: cannot read double length headers\n"); fflush(stdout); return;
        }
        if (len1 != len2) {
            printf("[FE][UCX][PreParse] ERROR: double length mismatch len1=%zu len2=%zu\n", len1, len2); fflush(stdout); return;
        }

        const unsigned char* str_ptr=nullptr;
        if (!read_bytes(len1, str_ptr)) {
            printf("[FE][UCX][PreParse] ERROR: not enough for routine string. need=%zu remain=%zu\n", len1, remain); fflush(stdout); return;
        }
        std::string routine_in_payload(reinterpret_cast<const char*>(str_ptr),
                                    len1 ? (len1 - 1) : 0);
        if (routine_in_payload != rname) {
            printf("[FE][UCX][PreParse] ERROR: routine mismatch: \"%s\" vs \"%s\"\n",
                routine_in_payload.c_str(), rname.c_str()); fflush(stdout); return;
        }

        size_t plen=0; const unsigned char* params_ptr=nullptr;
        if (!read_size_t(plen) || !read_bytes(plen, params_ptr)) {
            printf("[FE][UCX][PreParse] ERROR: cannot read params body\n"); fflush(stdout); return;
        }

        if (remain != 0) {
            printf("[FE][UCX][PreParse] WARN: trailing bytes remain=%zu\n", remain); fflush(stdout);
        }

        printf("[FE][UCX][PreParse] OK. routine=\"%s\" param_len=%zu (double-len)\n",
            routine_in_payload.c_str(), plen);
        print_hex_prefix("[FE][UCX][PreParse] params head", params_ptr, (plen<32?plen:32), 32);
        }


        // ------- Step 2: 发送 -------
        const bool expect_response = isSyncRoutine(rname);
        auto t0 = steady_clock::now();
        auto res = ucx->SubmitRequest("execute_routine",
                                      request_packet.GetBuffer(),
                                      request_packet.GetBufferSize(),
                                      expect_response);
        auto t1 = steady_clock::now();

        double send_wall_s = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
        frontend->mDataSent    += request_packet.GetBufferSize();
        frontend->mSendingTime += send_wall_s;

        printf("[FE][UCX] SubmitRequest done. expect_response=%s wall_send_s=%.6f\n",
               expect_response ? "true" : "false", send_wall_s);
        fflush(stdout);
        
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // 关键改动：异步 (expect_response==false) 直接返回，不解析响应
        if (!expect_response) {
            // 不读取 res.out、不触碰 mpOutputBuffer，不修改接收/执行时间统计
            return;
        }
        // ------- Step 3: 同步响应解析（修复点：用 vector<char> 适配 Buffer(char*, size_t)） -------
        // 先拿到原始响应
        const unsigned char* raw_rbuf = reinterpret_cast<const unsigned char*>(res.out.data());
        size_t raw_rlen = res.out.size();

        print_hex_prefix("[FE][UCX] raw response head", raw_rbuf, raw_rlen, 64);

        // 探测并去掉可选的“前置 size_t = payload_len”
        size_t offset = 0;
        if (raw_rlen >= sizeof(size_t)) {
            size_t pretended_len = 0;
            ::memcpy(&pretended_len, raw_rbuf, sizeof(size_t));
            if (pretended_len == (raw_rlen - sizeof(size_t))) {
                offset = sizeof(size_t);
                printf("[FE][UCX] Detected leading size_t payload_len in response, stripping %zu bytes.\n", offset);
                fflush(stdout);
            }
        }

        // 构造 resp_copy（真正要解析的主体）
        std::vector<char> resp_copy;
        if (raw_rlen >= offset) {
            resp_copy.assign(res.out.begin() + static_cast<std::ptrdiff_t>(offset), res.out.end());
        }
        if (resp_copy.empty()) {
            printf("[FE][UCX] ERROR: Empty response after stripping offset=%zu.\n", offset);
            fflush(stdout);
            return;
        }

        print_hex_prefix("[FE][UCX] response (post-strip) head", resp_copy.data(), resp_copy.size(), 64);

                // === 新：用裸指针解析响应，避免 Buffer::Get<char>(n) 额外吃一个 size_t ===
        {
            const unsigned char* p = reinterpret_cast<const unsigned char*>(resp_copy.data());
            size_t remain = resp_copy.size();

            // 自检：整体 payload 头
            print_hex_prefix("[FE][UCX][ParseRaw] resp_copy head", p, remain, 64);

            auto take = [&](void* dst, size_t n) -> bool {
                if (remain < n) return false;
                ::memcpy(dst, p, n);
                p += n; remain -= n;
                return true;
            };

            int exit_code = 0;
            double server_exec_sec = 0.0;
            size_t output_len = 0;

            // 解析头三项：int / double / size_t
            if (!take(&exit_code, sizeof(exit_code))) {
                printf("[FE][UCX][ParseRaw] ERROR: not enough for exit_code. remain=%zu\n", remain); fflush(stdout);
                return;
            }
            if (!take(&server_exec_sec, sizeof(server_exec_sec))) {
                printf("[FE][UCX][ParseRaw] ERROR: not enough for server_exec_sec. remain=%zu\n", remain); fflush(stdout);
                return;
            }
            if (!take(&output_len, sizeof(output_len))) {
                printf("[FE][UCX][ParseRaw] ERROR: not enough for out_len. remain=%zu\n", remain); fflush(stdout);
                return;
            }

            printf("[FE][UCX][ParseRaw] header parsed: exit_code=%d server_exec=%.6f out_len=%zu remain=%zu\n",
                exit_code, server_exec_sec, output_len, remain);
            fflush(stdout);

            // 解析 out 区域：直接切片，不再用 Buffer::Get<char>(n)
            frontend->mpOutputBuffer->Reset();
            if (output_len > 0) {
                if (remain < output_len) {
                    printf("[FE][UCX][ParseRaw] ERROR: not enough bytes for out body. need=%zu remain=%zu\n",
                        output_len, remain);
                    fflush(stdout);
                    return;
                }
                print_hex_prefix("[FE][UCX][ParseRaw] out head", p, (output_len < 32 ? output_len : 32), 32);

                // 直接 append
                frontend->mpOutputBuffer->Append(reinterpret_cast<const char*>(p), output_len);
                p += output_len;
                remain -= output_len;

                // 自检：是否还有尾巴
                if (remain != 0) {
                    printf("[FE][UCX][ParseRaw] WARN: trailing bytes after out. remain=%zu\n", remain);
                    fflush(stdout);
                }
            }

            // 统计与总耗时
            frontend->mExitCode = exit_code;
            frontend->mRoutineExecutionTime += server_exec_sec;
            frontend->mDataReceived += output_len;

            double wall_time = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
            printf("[FE] SYNC done: routine=\"%s\" exit_code=%d server_exec=%.6f s wall=%.6f s in=%zu B out=%zu B pid=%d tid=%d\n",
                rname.c_str(), exit_code, server_exec_sec, wall_time,
                param_len, output_len, (int)pid, (int)tid);
            fflush(stdout);
        }

        return;
    }
    // ================== OLD LOGICL PATH, NOT USE IN UCX ==================
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