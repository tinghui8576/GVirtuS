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

#include "CurandFrontend.h"

using namespace std;

static std::mutex generator_type_mutex; // Mutex to protect access to generator_is_host_map
static std::unordered_map<curandGenerator_t, bool> generator_is_host_map;  // true if host, false if device

/* Helper Functions */

bool isHostGenerator(curandGenerator_t generator) {
    std::lock_guard<std::mutex> lock(generator_type_mutex);
    auto it = generator_is_host_map.find(generator);
    if (it != generator_is_host_map.end()) {
        return it->second;
    }
    // Default if unknown, assume device generator
    return false;
}

/* HOST API */

extern "C" curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) {
    CurandFrontend::Prepare();
    CurandFrontend::AddVariableForArguments<int>(rng_type);
    CurandFrontend::Execute("curandCreateGenerator");
    if (CurandFrontend::Success()) {
        *generator = (curandGenerator_t)CurandFrontend::GetOutputVariable<uintptr_t>();
        
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map[*generator] = false;  // device generator
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t CURANDAPI curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) {
    CurandFrontend::Prepare();
    CurandFrontend::AddVariableForArguments<int>(rng_type);
    CurandFrontend::Execute("curandCreateGeneratorHost");
    if (CurandFrontend::Success()) {
        *generator = (curandGenerator_t)CurandFrontend::GetOutputVariable<uintptr_t>();
        
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map[*generator] = true;  // host generator
    }
    return CurandFrontend::GetExitCode();
}


extern "C" curandStatus_t curandGenerate( curandGenerator_t generator, unsigned int *outputPtr, size_t num){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    
    CurandFrontend::Execute("curandGenerate");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateLongLong( curandGenerator_t generator, unsigned long long *outputPtr, size_t num){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    
    CurandFrontend::Execute("curandGenerateLongLong");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num) {
    CurandFrontend::Prepare();
    CurandFrontend::AddVariableForArguments<uintptr_t>((uintptr_t)generator);

    bool is_host = isHostGenerator(generator);
    if (is_host) {
        CurandFrontend::AddHostPointerForArguments<float>(outputPtr, num); // Send serialized data
    } else {
        CurandFrontend::AddDevicePointerForArguments(outputPtr); // Send pointer address
    }

    CurandFrontend::AddVariableForArguments<size_t>(num);
    CurandFrontend::Execute("curandGenerateUniform");

    if (is_host && CurandFrontend::Success()) {
        float* backend_output = CurandFrontend::GetOutputHostPointer<float>(num);
        std::memcpy(outputPtr, backend_output, sizeof(float) * num);
    }

    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateNormal( curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(n);
    CurandFrontend::AddVariableForArguments<float>(mean);
    CurandFrontend::AddVariableForArguments<float>(stddev);
    
    CurandFrontend::Execute("curandGenerateNormal");
    return CurandFrontend::GetExitCode();
}


extern "C" curandStatus_t curandGenerateLogNormal( curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(n);
    CurandFrontend::AddVariableForArguments<float>(mean);
    CurandFrontend::AddVariableForArguments<float>(stddev);
    
    CurandFrontend::Execute("curandGenerateLogNormal");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGeneratePoisson( curandGenerator_t generator, unsigned int *outputPtr, size_t n, double lambda){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(n);
    CurandFrontend::AddVariableForArguments<double>(lambda);
    
    CurandFrontend::Execute("curandGeneratePoisson");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateUniformDouble( curandGenerator_t generator, double *outputPtr, size_t num){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    
    CurandFrontend::Execute("curandGenerateUniformDouble");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateNormalDouble( curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(n);
    CurandFrontend::AddVariableForArguments<double>(mean);
    CurandFrontend::AddVariableForArguments<double>(stddev);
    
    CurandFrontend::Execute("curandGenerateNormalDouble");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateLogNormalDouble( curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(n);
    CurandFrontend::AddVariableForArguments<double>(mean);
    CurandFrontend::AddVariableForArguments<double>(stddev);
    
    CurandFrontend::Execute("curandGenerateLogNormalDouble");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t CURANDAPI curandSetPseudoRandomGeneratorSeed( curandGenerator_t generator, unsigned long long seed ){
    CurandFrontend::Prepare();
    cout<<"ciao ciao ciao"<<endl;
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddVariableForArguments<unsigned long long>(seed);
    cout<<"Generator : "<<generator<<" seed: "<<seed<<endl;
    CurandFrontend::Execute("curandSetPseudoRandomGeneratorSeed");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t generator) {
    CurandFrontend::Prepare();
    CurandFrontend::AddVariableForArguments<uintptr_t>((uintptr_t)generator);
    CurandFrontend::Execute("curandDestroyGenerator");

    if (CurandFrontend::Success()) {
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map.erase(generator);
    }

    return CurandFrontend::GetExitCode();
}

/* --- HOST API END --- */