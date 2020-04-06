#include <iostream>
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "scratchpad.hpp"

namespace mkldnn {
namespace impl {

/* Allocating memory buffers on a page boundary to reduce TLB/page misses */
const size_t page_size = 2097152;

/*
  Implementation of the scratchpad_t interface that is compatible with
  a concurrent execution
*/
struct concurent_scratchpad_t : public scratchpad_t {
    concurent_scratchpad_t(size_t size) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/scratchpad.cpp:      concurent_scratchpad_t(size_t size) {" << std::endl;
        size_ = size;
        scratchpad_ = (char *) malloc(size, page_size);
        assert(scratchpad_ != nullptr);
    }

    ~concurent_scratchpad_t() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/scratchpad.cpp:      ~concurent_scratchpad_t() {" << std::endl;
        free(scratchpad_);
    }

    virtual char *get() const {
        return scratchpad_;
    }

private:
    char *scratchpad_;
    size_t size_;
};

/*
  Implementation of the scratchpad_t interface that uses a global
  scratchpad
*/

struct global_scratchpad_t : public scratchpad_t {
    global_scratchpad_t(size_t size) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/scratchpad.cpp:      global_scratchpad_t(size_t size) {" << std::endl;
        if (size > size_) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/scratchpad.cpp:          if (size > size_) {" << std::endl;
            if (scratchpad_ != nullptr) free(scratchpad_);
            size_ = size;
            scratchpad_ = (char *) malloc(size, page_size);
            assert(scratchpad_ != nullptr);
        }
        reference_count_++;
    }

    ~global_scratchpad_t() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/scratchpad.cpp:      ~global_scratchpad_t() {" << std::endl;
        reference_count_--;
        if (reference_count_ == 0) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/scratchpad.cpp:          if (reference_count_ == 0) {" << std::endl;
            free(scratchpad_);
            scratchpad_ = nullptr;
            size_ = 0;
        }
    }

    virtual char *get() const {
        return scratchpad_;
    }

private:
    thread_local static char *scratchpad_;
    thread_local static size_t size_;
    thread_local static unsigned int reference_count_;
};

thread_local char *global_scratchpad_t::scratchpad_ = nullptr;
thread_local size_t global_scratchpad_t::size_ = 0;
thread_local unsigned int global_scratchpad_t::reference_count_ = 0;


/*
   Scratchpad creation routine
*/
scratchpad_t *create_scratchpad(size_t size) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/scratchpad.cpp:  scratchpad_t *create_scratchpad(size_t size) {" << std::endl;
#ifndef MKLDNN_ENABLE_CONCURRENT_EXEC
    return new global_scratchpad_t(size);
#else
    return new concurent_scratchpad_t(size);
#endif
}

}
}
