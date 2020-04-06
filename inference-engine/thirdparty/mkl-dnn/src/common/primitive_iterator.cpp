#include <iostream>
/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#include <assert.h>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "primitive_iterator.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

status_t mkldnn_primitive_desc_iterator_create_v2(
        primitive_desc_iterator_t **iterator, const_c_op_desc_t c_op_desc,
        const primitive_attr_t *attr, engine_t *engine,
        const primitive_desc_t *hint_fwd_pd) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp:          const primitive_desc_t *hint_fwd_pd) {" << std::endl;
    const op_desc_t *op_desc = (const op_desc_t *)c_op_desc;

    auto it = new primitive_desc_iterator_t(engine, op_desc, attr, hint_fwd_pd);
    if (it == nullptr) return out_of_memory;

    ++(*it);
    if (*it == it->end()) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp:      if (*it == it->end()) {" << std::endl;
        delete it;
        return unimplemented;
    }

    *iterator = it;
    return success;
}

status_t mkldnn_primitive_desc_iterator_create(
        primitive_desc_iterator_t **iterator,
        const_c_op_desc_t c_op_desc, engine_t *engine,
        const primitive_desc_t *hint_fwd_pd) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp:          const primitive_desc_t *hint_fwd_pd) {" << std::endl;
    return mkldnn_primitive_desc_iterator_create_v2(iterator, c_op_desc,
            nullptr, engine, hint_fwd_pd);
}

status_t mkldnn_primitive_desc_iterator_next(
        primitive_desc_iterator_t *iterator) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp:          primitive_desc_iterator_t *iterator) {" << std::endl;
    if (iterator == nullptr) return invalid_arguments;
    ++(*iterator);
    return *iterator == iterator->end() ? iterator_ends : success;
}

primitive_desc_t *mkldnn_primitive_desc_iterator_fetch(
        const primitive_desc_iterator_t *iterator) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp:          const primitive_desc_iterator_t *iterator) {" << std::endl;
    if (iterator == nullptr) return nullptr;
    return *(*iterator);
}

status_t mkldnn_primitive_desc_clone(primitive_desc_t **primitive_desc,
        const primitive_desc_t *existing_primitive_desc) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp:          const primitive_desc_t *existing_primitive_desc) {" << std::endl;
    if (utils::any_null(primitive_desc, existing_primitive_desc))
        return invalid_arguments;
    return safe_ptr_assign<primitive_desc_t>(*primitive_desc,
            existing_primitive_desc->clone());
}

status_t mkldnn_primitive_desc_iterator_destroy(
        primitive_desc_iterator_t *iterator) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp:          primitive_desc_iterator_t *iterator) {" << std::endl;
    if (iterator != nullptr)
        delete iterator;
    return success;
}

status_t mkldnn_primitive_desc_create_v2(primitive_desc_t **primitive_desc,
        const_c_op_desc_t c_op_desc, const primitive_attr_t *attr,
        engine_t *engine, const primitive_desc_t *hint_fwd_pd) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp:          engine_t *engine, const primitive_desc_t *hint_fwd_pd) {" << std::endl;
    const op_desc_t *op_desc = (const op_desc_t *)c_op_desc;

    mkldnn_primitive_desc_iterator it(engine, op_desc, attr, hint_fwd_pd);
    ++it;
    if (it == it.end()) return unimplemented;

    return safe_ptr_assign<primitive_desc_t>(*primitive_desc, *it);
}

status_t mkldnn_primitive_desc_create(primitive_desc_t **primitive_desc,
        const_c_op_desc_t c_op_desc, engine_t *engine,
        const primitive_desc_t *hint_fwd_pd) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/primitive_iterator.cpp:          const primitive_desc_t *hint_fwd_pd) {" << std::endl;
    return mkldnn_primitive_desc_create_v2(primitive_desc, c_op_desc, nullptr,
            engine, hint_fwd_pd);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
