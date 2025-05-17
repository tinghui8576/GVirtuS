/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

#include <iostream>
#include <cstdio>
#include <string>
#include <mutex>
#include <unordered_map>

#include "CurandHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

static std::mutex generator_type_mutex;
static std::unordered_map<curandGenerator_t, bool> generator_is_host_map;

bool isHostGenerator(curandGenerator_t generator) {
    std::lock_guard<std::mutex> lock(generator_type_mutex);
    auto it = generator_is_host_map.find(generator);
    if (it != generator_is_host_map.end()) {
        return it->second;
    }
    return false;  // default to device generator
}

CURAND_ROUTINE_HANDLER(CreateGenerator) {
    // Create the generator, get handle
    curandGenerator_t generator;
    curandRngType_t gnrType = (curandRngType_t)in->Get<int>();
    curandStatus_t cs = curandCreateGenerator(&generator, gnrType);

    if (cs == CURAND_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map[generator] = false;  // device generator
    }

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add<uintptr_t>((uintptr_t)generator);
    return std::make_shared<Result>(cs, out);
}

CURAND_ROUTINE_HANDLER(CreateGeneratorHost) {
    // Create host generator
    curandGenerator_t generator;
    curandRngType_t gnrType = (curandRngType_t)in->Get<int>();
    curandStatus_t cs = curandCreateGeneratorHost(&generator, gnrType);

    if (cs == CURAND_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map[generator] = true;  // host generator
    }

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add<uintptr_t>((uintptr_t)generator);
    return std::make_shared<Result>(cs, out);
}

CURAND_ROUTINE_HANDLER(SetPseudoRandomGeneratorSeed) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetPseudoRandomGeneratorSeed"));

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    unsigned long long seed = in->Get<unsigned long long>();

    curandStatus_t cs = curandSetPseudoRandomGeneratorSeed(generator, seed);
    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(Generate) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Generate"));

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    bool is_host = isHostGenerator(generator);
    size_t num = 0;

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    curandStatus_t cs;

    if (is_host) {
        num = in->Get<size_t>();
        unsigned int* input_buffer = in->Assign<unsigned int>(num);  // discard input

        unsigned int* generated = new unsigned int[num];
        cs = curandGenerate(generator, generated, num);

        if (cs == CURAND_STATUS_SUCCESS) {
            out->Add(generated, num);
        }

        delete[] generated;
        return std::make_shared<Result>(cs, out);
    } else {
        uint64_t ptr_val = in->Get<uint64_t>();
        unsigned int* outputPtr = reinterpret_cast<unsigned int*>(ptr_val);
        num = in->Get<size_t>();

        cs = curandGenerate(generator, outputPtr, num);
        return std::make_shared<Result>(cs);
    }
}

CURAND_ROUTINE_HANDLER(GenerateLongLong) {
    std::cout << "[GenerateLongLong] Handler invoked" << std::endl;

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    std::cout << "[GenerateLongLong] Generator pointer: " << (uintptr_t)generator << std::endl;

    bool is_host = isHostGenerator(generator);
    std::cout << "[GenerateLongLong] is_host: " << (is_host ? "true" : "false") << std::endl;

    size_t num = 0;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    curandStatus_t cs;

    if (is_host) {
        num = in->Get<size_t>();
        std::cout << "[GenerateLongLong] Number of elements (host): " << num << std::endl;

        unsigned long long* input_buffer = in->Assign<unsigned long long>(num);  // discard input
        std::cout << "[GenerateLongLong] Discarded input_buffer pointer: " << (uintptr_t)input_buffer << std::endl;

        unsigned long long* generated = new unsigned long long[num];
        cs = curandGenerateLongLong(generator, generated, num);

        std::cout << "[GenerateLongLong] curandGenerateLongLong returned status: " << cs << std::endl;

        if (cs == CURAND_STATUS_SUCCESS) {
            out->Add(generated, num);
            std::cout << "[GenerateLongLong] Added generated data to output buffer" << std::endl;
        }

        delete[] generated;
        return std::make_shared<Result>(cs, out);
    } else {
        uint64_t ptr_val = in->Get<uint64_t>();
        unsigned long long* outputPtr = reinterpret_cast<unsigned long long*>(ptr_val);
        std::cout << "[GenerateLongLong] Received device pointer (uint64): " << ptr_val << std::endl;
        std::cout << "[GenerateLongLong] Output pointer reinterpret cast: " << (uintptr_t)outputPtr << std::endl;

        num = in->Get<size_t>();
        std::cout << "[GenerateLongLong] Number of elements (device): " << num << std::endl;

        cs = curandGenerateLongLong(generator, outputPtr, num);
        std::cout << "[GenerateLongLong] curandGenerateLongLong returned status: " << cs << std::endl;

        return std::make_shared<Result>(cs);
    }
}

CURAND_ROUTINE_HANDLER(GenerateUniform) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateUniform"));

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    bool is_host = isHostGenerator(generator);
    size_t num = 0;

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    curandStatus_t cs;

    if (is_host) {
        num = in->Get<size_t>();
        float* input_buffer = in->Assign<float>(num);  // we discard the input content

        float* generated = new float[num];
        cs = curandGenerateUniform(generator, generated, num);

        if (cs == CURAND_STATUS_SUCCESS) {
            out->Add(generated, num);  // Serialize result to send to frontend
        }

        delete[] generated;
        return std::make_shared<Result>(cs, out);
    } else {
        uint64_t ptr_val = in->Get<uint64_t>();
        float* outputPtr = reinterpret_cast<float*>(ptr_val);
        num = in->Get<size_t>();

        cs = curandGenerateUniform(generator, outputPtr, num);
        return std::make_shared<Result>(cs);  // No output buffer needed
    }
}

CURAND_ROUTINE_HANDLER(GenerateNormal) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateNormal"));

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    bool is_host = isHostGenerator(generator);

    size_t num = 0;
    float mean = 0.0f, stddev = 0.0f;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    curandStatus_t cs;

    if (is_host) {
        num = in->Get<size_t>();
        float* input_buffer = in->Assign<float>(num);  // discard input
        mean = in->Get<float>();
        stddev = in->Get<float>();

        float* generated = new float[num];
        cs = curandGenerateNormal(generator, generated, num, mean, stddev);

        if (cs == CURAND_STATUS_SUCCESS) {
            out->Add(generated, num);
        }

        delete[] generated;
        return std::make_shared<Result>(cs, out);
    } else {
        uint64_t ptr_val = in->Get<uint64_t>();
        float* outputPtr = reinterpret_cast<float*>(ptr_val);
        num = in->Get<size_t>();
        mean = in->Get<float>();
        stddev = in->Get<float>();

        cs = curandGenerateNormal(generator, outputPtr, num, mean, stddev);
        return std::make_shared<Result>(cs);
    }
}

CURAND_ROUTINE_HANDLER(GenerateLogNormal) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateLogNormal"));

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    bool is_host = isHostGenerator(generator);

    size_t num = 0;
    float mean = 0.0f, stddev = 0.0f;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    curandStatus_t cs;

    if (is_host) {
        num = in->Get<size_t>();
        float* input_buffer = in->Assign<float>(num);  // discard input
        mean = in->Get<float>();
        stddev = in->Get<float>();

        float* generated = new float[num];
        cs = curandGenerateLogNormal(generator, generated, num, mean, stddev);

        if (cs == CURAND_STATUS_SUCCESS) {
            out->Add(generated, num);
        }

        delete[] generated;
        return std::make_shared<Result>(cs, out);
    } else {
        uint64_t ptr_val = in->Get<uint64_t>();
        float* outputPtr = reinterpret_cast<float*>(ptr_val);
        num = in->Get<size_t>();
        mean = in->Get<float>();
        stddev = in->Get<float>();

        cs = curandGenerateLogNormal(generator, outputPtr, num, mean, stddev);
        return std::make_shared<Result>(cs);
    }
}

CURAND_ROUTINE_HANDLER(GeneratePoisson) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GeneratePoisson"));

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    bool is_host = isHostGenerator(generator);
    size_t num = 0;
    double lambda = 0.0;

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    curandStatus_t cs;

    if (is_host) {
        num = in->Get<size_t>();
        unsigned int* input_buffer = in->Assign<unsigned int>(num); // discard input
        lambda = in->Get<double>();

        unsigned int* generated = new unsigned int[num];
        cs = curandGeneratePoisson(generator, generated, num, lambda);

        if (cs == CURAND_STATUS_SUCCESS) {
            out->Add(generated, num);
        }

        delete[] generated;
        return std::make_shared<Result>(cs, out);
    } else {
        uint64_t ptr_val = in->Get<uint64_t>();
        unsigned int* outputPtr = reinterpret_cast<unsigned int*>(ptr_val);
        num = in->Get<size_t>();
        lambda = in->Get<double>();

        cs = curandGeneratePoisson(generator, outputPtr, num, lambda);
        return std::make_shared<Result>(cs);
    }
}

CURAND_ROUTINE_HANDLER(GenerateUniformDouble) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateUniformDouble"));

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    bool is_host = isHostGenerator(generator);
    size_t num = 0;

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    curandStatus_t cs;

    if (is_host) {
        num = in->Get<size_t>();
        double* input_buffer = in->Assign<double>(num); // discard input
        double* generated = new double[num];
        cs = curandGenerateUniformDouble(generator, generated, num);

        if (cs == CURAND_STATUS_SUCCESS) {
            out->Add(generated, num);
        }

        delete[] generated;
        return std::make_shared<Result>(cs, out);
    } else {
        uint64_t ptr_val = in->Get<uint64_t>();
        double* outputPtr = reinterpret_cast<double*>(ptr_val);
        num = in->Get<size_t>();

        cs = curandGenerateUniformDouble(generator, outputPtr, num);
        return std::make_shared<Result>(cs);
    }
}

CURAND_ROUTINE_HANDLER(GenerateNormalDouble) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateNormalDouble"));

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    bool is_host = isHostGenerator(generator);
    size_t num = 0;
    double mean, stddev;

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    curandStatus_t cs;

    if (is_host) {
        num = in->Get<size_t>();
        double* input_buffer = in->Assign<double>(num); // discard input
        mean = in->Get<double>();
        stddev = in->Get<double>();

        double* generated = new double[num];
        cs = curandGenerateNormalDouble(generator, generated, num, mean, stddev);

        if (cs == CURAND_STATUS_SUCCESS) {
            out->Add(generated, num);
        }

        delete[] generated;
        return std::make_shared<Result>(cs, out);
    } else {
        uint64_t ptr_val = in->Get<uint64_t>();
        double* outputPtr = reinterpret_cast<double*>(ptr_val);
        num = in->Get<size_t>();
        mean = in->Get<double>();
        stddev = in->Get<double>();

        cs = curandGenerateNormalDouble(generator, outputPtr, num, mean, stddev);
        return std::make_shared<Result>(cs);
    }
}

CURAND_ROUTINE_HANDLER(GenerateLogNormalDouble) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateLogNormalDouble"));

    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    bool is_host = isHostGenerator(generator);
    size_t num = 0;
    double mean, stddev;

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    curandStatus_t cs;

    if (is_host) {
        num = in->Get<size_t>();
        double* input_buffer = in->Assign<double>(num); // discard input buffer
        mean = in->Get<double>();
        stddev = in->Get<double>();

        double* generated = new double[num];
        cs = curandGenerateLogNormalDouble(generator, generated, num, mean, stddev);

        if (cs == CURAND_STATUS_SUCCESS) {
            out->Add(generated, num); // serialize to frontend
        }

        delete[] generated;
        return std::make_shared<Result>(cs, out);
    } else {
        uint64_t ptr_val = in->Get<uint64_t>();
        double* outputPtr = reinterpret_cast<double*>(ptr_val);
        num = in->Get<size_t>();
        mean = in->Get<double>();
        stddev = in->Get<double>();

        cs = curandGenerateLogNormalDouble(generator, outputPtr, num, mean, stddev);
        return std::make_shared<Result>(cs);
    }
}

CURAND_ROUTINE_HANDLER(DestroyGenerator) {
    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();
    curandStatus_t cs = curandDestroyGenerator(generator);

    if (cs == CURAND_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map.erase(generator);
    }

    return std::make_shared<Result>(cs);
}