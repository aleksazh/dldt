#include <iostream>
/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <string.h>

#define XBYAK64
#define XBYAK_NO_OP_NAMES

#include <cpu/xbyak/xbyak_util.h>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#endif

#include "mkldnn.h"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn.h"

#if defined(MKLDNN_X86_64)
#include "xmmintrin.h"
#endif

namespace mkldnn {
namespace impl {

int mkldnn_getenv(char *value, const char *name, int length) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  int mkldnn_getenv(char *value, const char *name, int length) {" << std::endl;
    int result = 0;
    int last_idx = 0;
    if (length > 1) {
        std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:      if (length > 1) {" << std::endl;
        int value_length = 0;
#ifdef _WIN32
        value_length = GetEnvironmentVariable(name, value, length);
        if (value_length >= length) {
            std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:          if (value_length >= length) {" << std::endl;
            result = -value_length;
        } else {
            last_idx = value_length;
            result = value_length;
        }
#else
        std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp: value: " << value << ", name: " << name << std::endl;
        char *buffer = getenv(name);
        if (buffer != NULL) {
            std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:          if (buffer != NULL) {" << std::endl;
            value_length = strlen(buffer);
            if (value_length >= length) {
                std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:              if (value_length >= length) {" << std::endl;
                result = -value_length;
            } else {
                strncpy(value, buffer, value_length);
                last_idx = value_length;
                result = value_length;
            }
        }
        std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp: (buffer == NULL) is true" << std::endl;
#endif
    }
    value[last_idx] = '\0';
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp: result: " << result << std::endl;
    return result;
}

static bool dump_jit_code;
static bool initialized;

bool mkldnn_jit_dump() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  bool mkldnn_jit_dump() {" << std::endl;
    if (!initialized) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:      if (!initialized) {" << std::endl;
        const int len = 2;
        char env_dump[len] = {0};
        dump_jit_code =
            mkldnn_getenv(env_dump, "MKLDNN_JIT_DUMP", len) == 1
            && atoi(env_dump) == 1;
        initialized = true;
    }
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp: dump_jit_code: " << dump_jit_code << std::endl;
    return dump_jit_code;
}

FILE *mkldnn_fopen(const char *filename, const char *mode) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  FILE *mkldnn_fopen(const char *filename, const char *mode) {" << std::endl;
#ifdef _WIN32
    FILE *fp = NULL;
    return fopen_s(&fp, filename, mode) ? NULL : fp;
#else
    return fopen(filename, mode);
#endif
}

thread_local unsigned int mxcsr_save;

void set_rnd_mode(round_mode_t rnd_mode) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  void set_rnd_mode(round_mode_t rnd_mode) {" << std::endl;
#if defined(MKLDNN_X86_64)
    mxcsr_save = _mm_getcsr();
    unsigned int mxcsr = mxcsr_save & ~(3u << 13);
    switch (rnd_mode) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:      switch (rnd_mode) {" << std::endl;
    case round_mode::nearest: mxcsr |= (0u << 13); break;
    case round_mode::down: mxcsr |= (1u << 13); break;
    default: assert(!"unreachable");
    }
    if (mxcsr != mxcsr_save) _mm_setcsr(mxcsr);
#else
    UNUSED(rnd_mode);
#endif
}

void restore_rnd_mode() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  void restore_rnd_mode() {" << std::endl;
#if defined(MKLDNN_X86_64)
    _mm_setcsr(mxcsr_save);
#endif
}

void *malloc(size_t size, int alignment) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  void *malloc(size_t size, int alignment) {" << std::endl;
    void *ptr;

#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
    int rc = ptr ? 0 : -1;
#else
    int rc = ::posix_memalign(&ptr, alignment, size);
#endif

    return (rc == 0) ? ptr : 0;
}

void free(void *p) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  void free(void *p) {" << std::endl;
#ifdef _WIN32
    _aligned_free(p);
#else
    ::free(p);
#endif
}

// Atomic operations
int32_t mkldnn_fetch_and_add(int32_t *dst, int32_t val) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  int32_t mkldnn_fetch_and_add(int32_t *dst, int32_t val) {" << std::endl;
#ifdef _WIN32
    return InterlockedExchangeAdd(reinterpret_cast<long*>(dst), val);
#else
    return __sync_fetch_and_add(dst, val);
#endif
}

static Xbyak::util::Cpu cpu_;

unsigned int get_cache_size(int level, bool per_core) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  unsigned int get_cache_size(int level, bool per_core) {" << std::endl;
    unsigned int l = level - 1;
    // Currently, if XByak is not able to fetch the cache topology
    // we default to 32KB of L1, 512KB of L2 and 1MB of L3 per core.
    if (cpu_.getDataCacheLevels() == 0){
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:      if (cpu_.getDataCacheLevels() == 0){" << std::endl;
        const int L1_cache_per_core = 32000;
        const int L2_cache_per_core = 512000;
        const int L3_cache_per_core = 1024000;
        int num_cores = per_core ? 1 : mkldnn_get_max_threads();
        switch(l){
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:          switch(l){" << std::endl;
            case(0): return L1_cache_per_core * num_cores;
            case(1): return L2_cache_per_core * num_cores;
            case(2): return L3_cache_per_core * num_cores;
            default: return 0;
        }
    }
    if (l < cpu_.getDataCacheLevels()) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:      if (l < cpu_.getDataCacheLevels()) {" << std::endl;
        return cpu_.getDataCacheSize(l)
               / (per_core ? cpu_.getCoresSharingDataCache(l) : 1);
    } else
        return 0;
}

}
}

mkldnn_status_t mkldnn_set_jit_dump(int dump) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  mkldnn_status_t mkldnn_set_jit_dump(int dump) {" << std::endl;
    using namespace mkldnn::impl::status;
    if (dump < 0) return invalid_arguments;
    mkldnn::impl::dump_jit_code = dump;
    mkldnn::impl::initialized = true;
    return success;
}

unsigned int mkldnn_get_cache_size(int level, int per_core) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/utils.cpp:  unsigned int mkldnn_get_cache_size(int level, int per_core) {" << std::endl;
    return mkldnn::impl::get_cache_size(level, per_core != 0);
}
