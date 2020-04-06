#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "system_allocator.hpp"

namespace InferenceEngine {

IAllocator* CreateDefaultAllocator() noexcept {
    try {
        return new SystemMemoryAllocator();
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/system_allocator.cpp:      } catch (...) {" << std::endl;
        return nullptr;
    }
}

}  // namespace InferenceEngine
