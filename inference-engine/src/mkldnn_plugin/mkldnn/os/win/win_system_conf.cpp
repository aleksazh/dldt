#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <windows.h>
#include <memory>
#include <vector>
#include "ie_parallel.hpp"

namespace MKLDNNPlugin {
namespace cpu {

int getNumberOfCPUCores() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/win/win_system_conf.cpp:  int getNumberOfCPUCores() {" << std::endl;
    const int fallback_val = parallel_get_max_threads();
    DWORD sz = 0;
    // querying the size of the resulting structure, passing the nullptr for the buffer
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &sz) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return fallback_val;

    std::unique_ptr<uint8_t[]> ptr(new uint8_t[sz]);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get()), &sz))
        return fallback_val;

    int phys_cores = 0;
    size_t offset = 0;
    do {
        offset += reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get() + offset)->Size;
        phys_cores++;
    } while (offset < sz);
    return phys_cores;
}

#if !(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
// OMP/SEQ threading on the Windows doesn't support NUMA
std::vector<int> getAvailableNUMANodes() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/win/win_system_conf.cpp:  std::vector<int> getAvailableNUMANodes() {" << std::endl; return std::vector<int>(1, 0); }
#endif

}  // namespace cpu
}  // namespace MKLDNNPlugin
