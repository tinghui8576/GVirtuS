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
#include <vector>

#define DEBUG

using gvirtus::backend::Process;
using gvirtus::common::LD_Lib;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::Endpoint;

using std::chrono::steady_clock;
using namespace std;

// NEW: Define the command header, must match the one in Frontend.cpp
#pragma pack(push, 1)
struct CommandHeader {
    uint8_t expect_response;
};
#pragma pack(pop)

Process::Process(std::shared_ptr<LD_Lib<Communicator, std::shared_ptr<Endpoint>>> communicator, vector <string> &plugins) : Observable() {
    // This function remains identical to your original version
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

// Using a compatible getstring that handles all communicator types correctly
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
                 std::string to_append = "libgvirtus-plugin-" + plug + ".so";
                 LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] appending " << to_append << ".");
                 auto ld_path = fs::path(gvirtus_home + "/lib").append(to_append);
                 try {
                     auto dl = std::make_shared<LD_Lib<Handler>>(ld_path, "create_t");
                     dl->build_obj();
                     _handlers.push_back(dl);
                 }
                 catch (const std::string &e) {
                     LOG4CPLUS_ERROR(logger, e);
                 }
             }
    );

    std::function<void(Communicator *)> execute = [this](Communicator *client_comm) {
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]" << " execute lambda called");
        string routine;
        std::shared_ptr<Buffer> input_buffer = std::make_shared<Buffer>();

        while (true) {
            // MODIFIED: Main loop now first reads the header
            CommandHeader hdr;
            try {
                if (client_comm->Read(reinterpret_cast<char*>(&hdr), sizeof(hdr)) != sizeof(hdr)) {
                    LOG4CPLUS_DEBUG(logger, "Client disconnected while reading header.");
                    break; 
                }
            } catch (const std::exception& e) {
                LOG4CPLUS_WARN(logger, "Failed to read command header, closing session: " << e.what());
                break;
            }

            if (!getstring(client_comm, routine)) {
                LOG4CPLUS_DEBUG(logger, "Client disconnected while reading routine name.");
                break;
            }
            
            LOG4CPLUS_DEBUG(logger, "✓ - Received routine " << routine << " (expect_response=" << (int)hdr.expect_response << ")");

            gvirtus::communicators::HybridCommunicator* hybrid = nullptr;
            if (client_comm && client_comm->to_string() == "hybridcommunicator") {
                hybrid = dynamic_cast<gvirtus::communicators::HybridCommunicator*>(client_comm);
            }
            if (hybrid) {
                const bool use_rdma =
                    routine.rfind("cudaRegisterFatBinary", 0)== 0 ||
                    routine.rfind("cudaRegisterFatBinaryEnd", 0)== 0 ||
                    routine.rfind("cudaMemcpyAsync", 0) == 0 ||   
                    routine.rfind("cudaMemcpy",      0) == 0;     
                if (use_rdma) {
                    hybrid->begin_call(routine, gvirtus::communicators::Transport::RDMA, 1);
                } else {
                    hybrid->begin_call(routine, gvirtus::communicators::Transport::TCP, 0);
                }
            }
            
            input_buffer->Reset(client_comm);

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
                result->TimeTaken(std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start).count() / 1000.0);
            }
            
            // MODIFIED: CRITICAL CHANGE - Only dump response if header says so
            if (hdr.expect_response) {
                result->Dump(client_comm);
            }

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
            Communicator *client = const_cast<Communicator *>(_communicator->obj_ptr()->Accept());
            
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
    }
    catch (std::string &exc) {
        LOG4CPLUS_ERROR(logger, "✖ - [Process " << getpid() << "]: " << exc);
    }

    LOG4CPLUS_DEBUG(logger, "✓ - Process::Start() returned [Process " << getpid() << "].");
}

Process::~Process() {
    _communicator.reset();
    _handlers.clear();
    mPlugins.clear();
}