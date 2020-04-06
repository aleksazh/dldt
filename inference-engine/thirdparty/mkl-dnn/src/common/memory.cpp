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
#include <stddef.h>
#include <stdint.h>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "memory_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::data_type;

namespace {
bool memory_desc_sanity_check(int ndims,const dims_t dims,
        data_type_t data_type, memory_format_t format) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          data_type_t data_type, memory_format_t format) {" << std::endl;
    if (ndims == 0) return true;

    bool ok = true
        && dims != nullptr
        && 0 < ndims && ndims <= TENSOR_MAX_DIMS
        && one_of(data_type, f32, s32, s16, bf16, s8, u8, bin)
        && format != memory_format::undef;
    if (!ok) return false;
    for (int d = 0; d < ndims; ++d)
        if (dims[d] < 0) return false;

    return true;
}

bool memory_desc_sanity_check(const memory_desc_t *md) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:  bool memory_desc_sanity_check(const memory_desc_t *md) {" << std::endl;
    if (md == nullptr) return false;
    return memory_desc_sanity_check(md->ndims, md->dims, md->data_type,
            md->format);
}
}

status_t mkldnn_memory_desc_init(memory_desc_t *memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, memory_format_t format) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          const dims_t dims, data_type_t data_type, memory_format_t format) {" << std::endl;
    if (any_null(memory_desc)) return invalid_arguments;
    if (ndims == 0 || format == memory_format::undef) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      if (ndims == 0 || format == memory_format::undef) {" << std::endl;
        *memory_desc = types::zero_md();
        return success;
    }

    /* memory_desc != 0 */
    bool args_ok = !any_null(memory_desc)
        && memory_desc_sanity_check(ndims, dims, data_type, format);
    if (!args_ok) return invalid_arguments;

    memory_desc_t md;
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.primitive_kind = primitive_kind::memory;
    md.data_type = data_type;
    md.format = format;

    status_t status = success;
    if (one_of(format, memory_format::undef, blocked, wino_fmt, rnn_packed)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      if (one_of(format, memory_format::undef, blocked, wino_fmt, rnn_packed)) {" << std::endl;
        status = invalid_arguments;
    } else if (format == any) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      } else if (format == any) {" << std::endl;
        // nop
    } else if (types::format_normalize(format) == blocked) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      } else if (types::format_normalize(format) == blocked) {" << std::endl;
        status = memory_desc_wrapper::compute_blocking(md);
    } else {
        assert(!"unreachable");
        status = invalid_arguments;
    }

    if (status == success)
        *memory_desc = md;

    return status;
}

status_t mkldnn_memory_primitive_desc_create(primitive_desc_t **memory_pd,
        const memory_desc_t *memory_desc, engine_t *engine) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          const memory_desc_t *memory_desc, engine_t *engine) {" << std::endl;
    bool args_ok = !any_null(memory_pd, memory_desc, engine)
        && memory_desc_sanity_check(memory_desc)
        && memory_desc_wrapper(*memory_desc).is_defined();
    if (!args_ok) return invalid_arguments;
    return engine->memory_primitive_desc_create(
            (memory_pd_t**)memory_pd, memory_desc);
}

status_t mkldnn_view_primitive_desc_create(primitive_desc_t **view_pd,
        const primitive_desc_t *memory_pd, const dims_t dims,
        const dims_t offsets) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          const dims_t offsets) {" << std::endl;
    const memory_pd_t *mpd =
        (const memory_pd_t*)memory_pd;

    bool args_ok = !any_null(view_pd, memory_pd, dims, offsets)
        && memory_pd->kind() == primitive_kind::memory
        && memory_desc_sanity_check(mpd->desc());
    if (!args_ok) return invalid_arguments;

    memory_desc_wrapper md(*mpd->desc());
    for (int d = 0; d < md.ndims(); ++d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      for (int d = 0; d < md.ndims(); ++d) {" << std::endl;
        if (dims[d] < 0 || offsets[d] < 0
                || (offsets[d] + dims[d] > md.dims()[d]))
            return invalid_arguments;
    }
    return memory_pd->engine()->view_primitive_desc_create(
            (view_pd_t**)view_pd, mpd, dims, offsets);
}

int mkldnn_memory_primitive_desc_equal(const primitive_desc_t *lhs,
        const primitive_desc_t *rhs) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          const primitive_desc_t *rhs) {" << std::endl;
    bool args_ok = !any_null(lhs, rhs)
        && lhs->engine() == rhs->engine()
        && one_of(lhs->kind(), primitive_kind::memory, primitive_kind::view)
        && one_of(rhs->kind(), primitive_kind::memory, primitive_kind::view);
    if (!args_ok) return 0;
    auto l = (const memory_pd_t *)lhs;
    auto r = (const memory_pd_t *)rhs;
    /* FIXME: view! */
    return l->is_equal(r);
}

size_t mkldnn_memory_primitive_desc_get_size(const primitive_desc_t *memory_pd)
{
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:  size_t mkldnn_memory_primitive_desc_get_size(const primitive_desc_t *memory_pd) {" << std::endl;
    bool args_ok = !any_null(memory_pd)
        && memory_pd->kind() == primitive_kind::memory;
    if (!args_ok) return 0;
    /* FIXME: view? */
    return ((memory_pd_t*)memory_pd)->get_size();
}

status_t mkldnn_memory_get_data_handle(const primitive_t *memory,
        void **handle) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          void **handle) {" << std::endl;
    if (any_null(handle))
        return invalid_arguments;
    if (memory == nullptr) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      if (memory == nullptr) {" << std::endl;
        *handle = nullptr;
        return success;
    }
    if (memory->kind() != primitive_kind::memory)
        return invalid_arguments;
    return memory->get_data_handle(handle);
}

status_t mkldnn_memory_set_data_handle(primitive_t *memory, void *handle) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:  status_t mkldnn_memory_set_data_handle(primitive_t *memory, void *handle) {" << std::endl;
    if (any_null(memory) || memory->kind() != primitive_kind::memory)
        return invalid_arguments;
    return memory->set_data_handle(handle, true);
}

status_t mkldnn_memory_set_data_handle_no_pads_proc(primitive_t *memory, void *handle) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:  status_t mkldnn_memory_set_data_handle_no_pads_proc(primitive_t *memory, void *handle) {" << std::endl;
    if (any_null(memory) || memory->kind() != primitive_kind::memory)
        return invalid_arguments;
    return memory->set_data_handle(handle, false);
}

status_t mkldnn_concat_primitive_desc_create_v2(primitive_desc_t **concat_pd,
        const memory_desc_t *output_d, int n, int concat_dim,
        const primitive_desc_t **input_pds, const primitive_attr_t *attr) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          const primitive_desc_t **input_pds, const primitive_attr_t *attr) {" << std::endl;
    bool args_ok = !any_null(concat_pd, input_pds) && n > 0;
    if (!args_ok) return invalid_arguments;
    for (int i = 0; i < n; ++i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      for (int i = 0; i < n; ++i) {" << std::endl;
        if (input_pds[i] == nullptr ||
                input_pds[i]->kind() != primitive_kind::memory)
            return invalid_arguments;
    }

    const primitive_attr_t dummy_attr;
    if (attr == NULL)
        attr = &dummy_attr;

    auto i_mpds = (const memory_pd_t **)input_pds;
    engine_t *engine = i_mpds[0]->engine();
    const int ndims = i_mpds[0]->desc()->ndims;
    const dims_t &dims = i_mpds[0]->desc()->dims;
    const data_type_t dt = i_mpds[0]->desc()->data_type;

    int concat_dim_sz = dims[concat_dim];
    for (int i = 1; i < n; ++i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      for (int i = 1; i < n; ++i) {" << std::endl;
        if (i_mpds[i]->engine() != engine) return invalid_arguments;
        if (i_mpds[i]->desc()->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          for (int d = 0; d < ndims; ++d) {" << std::endl;
            if (d == concat_dim) continue;
            if (i_mpds[i]->desc()->dims[d] != dims[d])
                return invalid_arguments;
        }
        if (i_mpds[i]->desc()->data_type != dt) return invalid_arguments;
        concat_dim_sz += i_mpds[i]->desc()->dims[concat_dim];
    }

    memory_desc_t dummy_output_d;
    if (output_d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      if (output_d) {" << std::endl;
        if (output_d->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          for (int d = 0; d < ndims; ++d) {" << std::endl;
            if (output_d->dims[d] !=
                    (d == concat_dim ? concat_dim_sz : dims[d]))
                return invalid_arguments;
        }
    } else {
        dummy_output_d = *i_mpds[0]->desc();
        dummy_output_d.dims[concat_dim] = concat_dim_sz;
        dummy_output_d.format = memory_format::any;
        output_d = &dummy_output_d;
    }

    auto c_pd = reinterpret_cast<concat_pd_t **>(concat_pd);

    for (auto c = engine->get_concat_implementation_list(); *c; ++c) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      for (auto c = engine->get_concat_implementation_list(); *c; ++c) {" << std::endl;
        if ((*c)(c_pd, output_d, n, concat_dim, i_mpds, attr) == success) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          if ((*c)(c_pd, output_d, n, concat_dim, i_mpds, attr) == success) {" << std::endl;
            (*c_pd)->init_info();
            return success;
        }
    }
    return unimplemented;
}

status_t mkldnn_concat_primitive_desc_create(primitive_desc_t **concat_pd,
        const memory_desc_t *output_d, int n, int concat_dim,
        const primitive_desc_t **input_pds) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          const primitive_desc_t **input_pds) {" << std::endl;
    return mkldnn_concat_primitive_desc_create_v2(concat_pd, output_d, n,
            concat_dim, input_pds, nullptr);
}

status_t mkldnn_sum_primitive_desc_create_v2(primitive_desc_t **sum_pd,
        const memory_desc_t *output_d, int n, const float *scales,
        const primitive_desc_t **input_pds, const primitive_attr_t *attr) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          const primitive_desc_t **input_pds, const primitive_attr_t *attr) {" << std::endl;
    bool args_ok = !any_null(sum_pd, input_pds, scales) && n > 0;
    if (!args_ok) return invalid_arguments;
    for (int i = 0; i < n; ++i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      for (int i = 0; i < n; ++i) {" << std::endl;
        if (input_pds[i] == nullptr ||
                input_pds[i]->kind() != primitive_kind::memory)
            return invalid_arguments;
    }

    const primitive_attr_t dummy_attr;
    if (attr == NULL)
        attr = &dummy_attr;

    auto i_mpds = (const memory_pd_t **)input_pds;
    engine_t *engine = i_mpds[0]->engine();
    const int ndims = i_mpds[0]->desc()->ndims;
    const dims_t &dims = i_mpds[0]->desc()->dims;
    const data_type_t dt = i_mpds[0]->desc()->data_type;

    for (int i = 1; i < n; ++i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      for (int i = 1; i < n; ++i) {" << std::endl;
        if (i_mpds[i]->engine() != engine) return invalid_arguments;
        if (i_mpds[i]->desc()->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          for (int d = 0; d < ndims; ++d) {" << std::endl;
            if (i_mpds[i]->desc()->dims[d] != dims[d])
                return invalid_arguments;
        }
        if (i_mpds[i]->desc()->data_type != dt) return invalid_arguments;
    }

    memory_desc_t dummy_output_d;
    if (output_d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      if (output_d) {" << std::endl;
        if (output_d->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          for (int d = 0; d < ndims; ++d) {" << std::endl;
            if (output_d->dims[d] != dims[d])
                return invalid_arguments;
        }
    } else {
        dummy_output_d = *i_mpds[0]->desc();
        dummy_output_d.format = memory_format::any;
        output_d = &dummy_output_d;
    }

    auto s_pd = reinterpret_cast<sum_pd_t **>(sum_pd);

    for (auto s = engine->get_sum_implementation_list(); *s; ++s) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:      for (auto s = engine->get_sum_implementation_list(); *s; ++s) {" << std::endl;
        if ((*s)(s_pd, output_d, n, scales, i_mpds, attr) == success) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          if ((*s)(s_pd, output_d, n, scales, i_mpds, attr) == success) {" << std::endl;
            (*s_pd)->init_info();
            return success;
        }
    }
    return unimplemented;
}

status_t mkldnn_sum_primitive_desc_create(primitive_desc_t **sum_pd,
        const memory_desc_t *output_d, int n, const float *scales,
        const primitive_desc_t **input_pds) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/common/memory.cpp:          const primitive_desc_t **input_pds) {" << std::endl;
    return mkldnn_sum_primitive_desc_create_v2(sum_pd, output_d, n, scales,
            input_pds, nullptr);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
