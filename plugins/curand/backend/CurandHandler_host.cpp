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
    out->Add<long long int>((long long int)generator);
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
    out->Add<long long int>((long long int)generator);
    return std::make_shared<Result>(cs, out);
}

CURAND_ROUTINE_HANDLER(Generate){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Generate"));
    
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    unsigned int * outputPtr = in->Assign<unsigned int>();
    size_t num = in->Get<size_t>();
    
    curandStatus_t cs = curandGenerate(generator,outputPtr,num);
    
    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(GenerateLongLong){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateLongLong"));
    
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    unsigned long long int * outputPtr = in->Assign<unsigned long long int>();
    size_t num = in->Get<size_t>();
    
    curandStatus_t cs = curandGenerateLongLong(generator,outputPtr,num);
    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(GenerateUniform) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateUniform"));

    // Read generator handle (type long long int or uintptr_t)
    curandGenerator_t generator = (curandGenerator_t)in->Get<uintptr_t>();

    // Now you must know whether this is a host or device pointer.
    // For simplicity, assume you track generator types similarly in backend:
    bool is_host = isHostGenerator(generator);

    float* outputPtr = nullptr;
    if (is_host) {
        // Host pointer: data is serialized in the buffer, so get actual float array
        outputPtr = in->Assign<float>();  // Deserialize float array data
    } else {
        // Device pointer: read pointer value (uint64_t) from buffer, cast to float*
        outputPtr = (float*)(uintptr_t)in->Get<uint64_t>();
    }

    size_t num = in->Get<size_t>();

    curandStatus_t cs = curandGenerateUniform(generator, outputPtr, num);

    // Send back generated values if host generator
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    if (is_host && cs == CURAND_STATUS_SUCCESS) {
        out->Add<float>(outputPtr, num);
    }

    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(GenerateNormal){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateNormal"));
    
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    float * outputPtr = in->Assign<float>();
    size_t n = in->Get<size_t>();
    float mean = in->Get<float>();
    float stddev = in->Get<float>();
    
    curandStatus_t cs = curandGenerateNormal(generator,outputPtr,n,mean,stddev);
    return std::make_shared<Result>(cs);
}


CURAND_ROUTINE_HANDLER(GenerateLogNormal){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateLogNormal"));
    
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    float * outputPtr = in->Assign<float>();
    size_t n = in->Get<size_t>();
    float mean = in->Get<float>();
    float stddev = in->Get<float>();
    
    curandStatus_t cs = curandGenerateLogNormal(generator,outputPtr,n,mean,stddev);
    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(GeneratePoisson){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GeneratePoisson"));
    
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    unsigned int * outputPtr = in->Assign<unsigned int>();
    size_t n = in->Get<size_t>();
    double lambda = in->Get<double>();
    
    curandStatus_t cs = curandGeneratePoisson(generator,outputPtr,n,lambda);
    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(GenerateUniformDouble){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateUniformDouble"));
    
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    double * outputPtr = in->Assign<double>();
    size_t num = in->Get<size_t>();
    
    curandStatus_t cs = curandGenerateUniformDouble(generator,outputPtr,num);
    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(GenerateNormalDouble){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateNormalDouble"));
    
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    double * outputPtr = in->Assign<double>();
    size_t n = in->Get<size_t>();
    double mean = in->Get<double>();
    double stddev = in->Get<double>();
    
    
    curandStatus_t cs = curandGenerateNormalDouble(generator,outputPtr,n,mean,stddev);
    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(GenerateLogNormalDouble){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateLogNormalDouble"));
    
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    double * outputPtr = in->Assign<double>();
    size_t n = in->Get<size_t>();
    double mean = in->Get<double>();
    double stddev = in->Get<double>();
    
    
    curandStatus_t cs = curandGenerateLogNormalDouble(generator,outputPtr,n,mean,stddev);
    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(SetPseudoRandomGeneratorSeed){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GenerateLogNormalDouble"));
    cout<<"ciao ciao ciao"<<endl;
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    unsigned long long seed = in->Get<unsigned long long>();
    cout<<"generator: "<< generator << " seed: "<< seed<<endl;
    curandStatus_t cs = curandSetPseudoRandomGeneratorSeed(generator,seed);
    return std::make_shared<Result>(cs);
}

CURAND_ROUTINE_HANDLER(DestroyGenerator) {
    curandGenerator_t generator = (curandGenerator_t)in->Get<long long int>();
    curandStatus_t cs = curandDestroyGenerator(generator);

    if (cs == CURAND_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map.erase(generator);
    }

    return std::make_shared<Result>(cs);
}