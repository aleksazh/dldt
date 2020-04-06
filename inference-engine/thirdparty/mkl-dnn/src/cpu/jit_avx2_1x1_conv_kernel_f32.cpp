#include <iostream>
/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_memory.hpp"

#include "jit_avx2_1x1_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

void jit_avx2_1x1_conv_kernel_f32::generate_bcast_loop(int load_loop_blk)
{
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:  void jit_avx2_1x1_conv_kernel_f32::generate_bcast_loop(int load_loop_blk) {" << std::endl;
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, reg_bcast_loop_work);

    Label bcast_loop, bcast_loop_tail;

    cmp(bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop); {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          for (int i = 0; i < num_substeps; i++) {" << std::endl;
            generate_reduce_loop(load_loop_blk, jcp.ur);
            if (i < num_substeps - 1) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              if (i < num_substeps - 1) {" << std::endl;
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_substep);
            } else {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step
                        - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_step
                        - (num_substeps - 1) * jcp.bcast_loop_output_substep);
            }
        }
        sub(bcast_loop_iter, jcp.bcast_block);
        cmp(bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      if (jcp.ur_tail) {" << std::endl;
        Label bcast_loop_tail_out;
        cmp(bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        generate_reduce_loop(load_loop_blk, jcp.ur_tail);
        L(bcast_loop_tail_out);
    }
}

void jit_avx2_1x1_conv_kernel_f32::generate_reduce_loop(
        int load_loop_blk, int ur)
{
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          int load_loop_blk, int ur) {" << std::endl;
    auto vreg_load = [=](int i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto vreg_load = [=](int i) {" << std::endl;
        return Ymm(ur * load_loop_blk + i);
    };

    auto vreg_accum = [=](int i, int j) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto vreg_accum = [=](int i, int j) {" << std::endl;
        return Ymm(j + i*ur);
    };

    auto bias_ptr = [=](int i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto bias_ptr = [=](int i) {" << std::endl;
        return ptr[reg_bias_data + sizeof(float) * jcp.oc_block * i];
    };

    auto bcast_ptr = [=](int u, int j) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto bcast_ptr = [=](int u, int j) {" << std::endl;
        assert(j < jcp.ur);
        assert(u <= jcp.reduce_loop_unroll);
        size_t offt;
        if (one_of(jcp.prop_kind,
                    forward_training, forward_inference, backward_data))
        {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                      forward_training, forward_inference, backward_data))         {" << std::endl;
            assert(jcp.reduce_loop_unroll == (jcp.prop_kind == backward_data)
                    ? jcp.oc_block : jcp.ic_block);
            auto height = (jcp.prop_kind == backward_data) ? jcp.os : jcp.is;
            offt = (u == jcp.reduce_loop_unroll)
                ? (height + j) * jcp.reduce_loop_unroll
                : j * jcp.reduce_loop_unroll + u;
        } else
            offt = u * jcp.ic_block + j;
        return ptr[aux_reg_bcast_data + sizeof(float) * offt];
    };

    auto load_ptr = [=](int u, int i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto load_ptr = [=](int u, int i) {" << std::endl;
        size_t offt;
        size_t u0 = u % jcp.reduce_loop_unroll;
        size_t u1 = u / jcp.reduce_loop_unroll;
        switch (jcp.prop_kind) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          switch (jcp.prop_kind) {" << std::endl;
        case backward_data:
            offt = (i * jcp.oc_block + u0) * jcp.ic_block;
            break;
        case backward_weights:
            offt = (i * jcp.os + u0) * jcp.oc_block;
            break;
        default:
            offt = (i * jcp.ic + u0) * jcp.oc_block;
        }
        return ptr[aux_reg_load_data
            + u1 * jcp.reduce_loop_load_step + sizeof(float) * offt];
    };

    auto output_ptr = [=](int i, int j) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto output_ptr = [=](int i, int j) {" << std::endl;
        switch (jcp.prop_kind) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          switch (jcp.prop_kind) {" << std::endl;
        case backward_data:
            return ptr[aux_reg_output_data +
                (i * jcp.is + j) * jcp.ic_block * sizeof(float)];
        case backward_weights:
            return ptr[aux_reg_output_data
                + (i ? reg_output_stride * i : 0) // TODO: Xbyak should allow 0 scale
                + sizeof(float) * jcp.oc_block * j];
        default:
            if (jcp.with_dw_conv) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              if (jcp.with_dw_conv) {" << std::endl;
                return ptr[aux_reg_output_data +
                           (i * jcp_dw.kh * jcp.ow + j) * jcp.oc_block * sizeof(float)];
            } else {
                return ptr[aux_reg_output_data +
                           (i * jcp.os + j) * jcp.oc_block * sizeof(float)];
            }
        }
    };

    auto init = [=]() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto init = [=]() {" << std::endl;
        Label init_done, init_zero;

        if (jcp.with_bias && one_of(jcp.prop_kind, forward_training,
                    forward_inference)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                      forward_inference)) {" << std::endl;
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jz(init_zero);

            for (int i = 0; i < load_loop_blk; i++)
                for (int j = 0; j < ur; ++j)
                    vmovups(vreg_accum(i, j), bias_ptr(i));
            jmp(init_done);
        }

        L(init_zero);
        for (int i = 0; i < load_loop_blk; ++i)
            for (int j = 0; j < ur; ++j) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              for (int j = 0; j < ur; ++j) {" << std::endl;
                auto r = vreg_accum(i, j);
                vxorps(r, r, r);
            }

        L(init_done);
        for (int i = 0; i < load_loop_blk; ++i)
            vmovups(vreg_load(i), load_ptr(0, i));
        vbroadcastss(vreg_bcast, bcast_ptr(0, 0));
    };

    auto store = [=]() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto store = [=]() {" << std::endl;
        Label store_noadd;

        if (!jcp.with_sum) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          if (!jcp.with_sum) {" << std::endl;
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jnz(store_noadd, T_NEAR);
        }

        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              for (int i = 0; i < load_loop_blk; ++i) {" << std::endl;
                auto r = vreg_accum(i, j);
                vaddps(r, r, output_ptr(i, j));
            }

        L(store_noadd);

        Label store_norelu;
        test(reg_reduce_pos_flag, FLAG_REDUCE_LAST);
        jz(store_norelu, T_NEAR);

        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        const auto &p = attr_.post_ops_;

        int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
        for (int i = 0; i < end_idx; i++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          for (int i = 0; i < end_idx; i++) {" << std::endl;
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              if (post_op.is_eltwise()) {" << std::endl;
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(0, ur * load_loop_blk);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              } else if (post_op.is_depthwise()) {" << std::endl;
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                for (int j = 0; j < load_loop_blk; ++j) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                  for (int j = 0; j < load_loop_blk; ++j) {" << std::endl;
                    int start_idx = vreg_accum(j, 0).getIdx();
                    int end_idx = start_idx + ur;

                    depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                            start_idx, end_idx, reg_d_weights, reg_d_bias);

                    add(reg_d_weights, jcp.oc_block * sizeof(float));
                    add(reg_d_bias, jcp.oc_block * sizeof(float));
                }

                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              } else if (post_op.is_quantization()) {" << std::endl;
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                for (int ii = 0; ii < load_loop_blk; ii++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                  for (int ii = 0; ii < load_loop_blk; ii++) {" << std::endl;
                    uni_vmovups(ymm_d_weights, ptr[reg_d_weights + ii * jcp.oc_block * sizeof(float)]);
                    uni_vmovups(ymm_d_bias, ptr[reg_d_bias + ii * jcp.oc_block * sizeof(float)]);

                    for (int jj = 0; jj < ur; jj++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                      for (int jj = 0; jj < ur; jj++) {" << std::endl;
                        Ymm ymm_dst = vreg_accum(ii, jj);

                        uni_vmaxps(ymm_dst, ymm_dst, ymm_d_weights);
                        uni_vminps(ymm_dst, ymm_dst, ymm_d_bias);
                    }
                }

                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                for (int ii = 0; ii < load_loop_blk; ii++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                  for (int ii = 0; ii < load_loop_blk; ii++) {" << std::endl;
                    uni_vmovups(ymm_d_weights, ptr[reg_d_weights + ii * jcp.oc_block * sizeof(float)]);
                    uni_vmovups(ymm_d_bias, ptr[reg_d_bias + ii * jcp.oc_block * sizeof(float)]);

                    for (int jj = 0; jj < ur; jj++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                      for (int jj = 0; jj < ur; jj++) {" << std::endl;
                        Ymm ymm_dst = vreg_accum(ii, jj);

                        uni_vfmadd213ps(ymm_dst, ymm_d_weights, ymm_d_bias);
                        uni_vroundps(ymm_dst, ymm_dst, 0);
                    }
                }

                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data));

                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);

                for (int ii = 0; ii < load_loop_blk; ii++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                  for (int ii = 0; ii < load_loop_blk; ii++) {" << std::endl;
                    uni_vmovups(ymm_d_weights, ptr[reg_d_weights + ii * jcp.oc_block * sizeof(float)]);
                    uni_vmovups(ymm_d_bias, ptr[reg_d_bias + ii * jcp.oc_block * sizeof(float)]);

                    for (int jj = 0; jj < ur; jj++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                      for (int jj = 0; jj < ur; jj++) {" << std::endl;
                        Ymm ymm_dst = vreg_accum(ii, jj);

                        uni_vfmadd213ps(ymm_dst, ymm_d_weights, ymm_d_bias);
                    }
                }
            }
        }

        L(store_norelu);

        for (int j = 0; j < ur; ++j)
            for (int i = 0; i < load_loop_blk; ++i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              for (int i = 0; i < load_loop_blk; ++i) {" << std::endl;
                vmovups(output_ptr(i, j), vreg_accum(i, j));
            }
    };

    auto fma_block = [=](bool last_block) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto fma_block = [=](bool last_block) {" << std::endl;
        for (int u = 0; u < jcp.reduce_loop_unroll; ++u) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          for (int u = 0; u < jcp.reduce_loop_unroll; ++u) {" << std::endl;
            for (int j = 0; j < ur; ++j) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              for (int j = 0; j < ur; ++j) {" << std::endl;
                for (int i = 0; i < load_loop_blk; ++i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:                  for (int i = 0; i < load_loop_blk; ++i) {" << std::endl;
                    if (mayiuse(avx2))
                        vfmadd231ps(vreg_accum(i, j), vreg_load(i), vreg_bcast);
                    else { // Intel(R) Advanced Vector Extensions (Intel(R) AVX) support
                        vmulps(vtmp, vreg_bcast, vreg_load(i));
                        vaddps(vreg_accum(i, j), vreg_accum(i, j), vtmp);
                    }
                    if (j == ur - 1 && !(last_block
                                && u == jcp.reduce_loop_unroll - 1))
                        vmovups(vreg_load(i), load_ptr(u + 1, i));
                }
                if (j < ur - 1)
                    vbroadcastss(vreg_bcast, bcast_ptr(u, j + 1));
            }
            if (!last_block || u < jcp.reduce_loop_unroll - 1)
                vbroadcastss(vreg_bcast, bcast_ptr(u + 1, 0));
        }
    };

    Label reduce_loop, reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);
    mov(aux_reg_bcast_data, aux1_reg_bcast_data);

    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop); {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
}

void jit_avx2_1x1_conv_kernel_f32::generate_diff_bias_loop(int load_loop_blk)
{
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:  void jit_avx2_1x1_conv_kernel_f32::generate_diff_bias_loop(int load_loop_blk) {" << std::endl;
    if (!jcp.with_bias || jcp.prop_kind != backward_weights)
        return;

    Label diff_bias_loop, diff_bias_loop_out, diff_bias_init_out;
    Label diff_bias_load;

    auto diff_bias_ptr = [=](int i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto diff_bias_ptr = [=](int i) {" << std::endl;
        return ptr[reg_diff_bias_data + i * jcp.oc_block * sizeof(float)];
    };

    auto load_ptr = [=](int u, int i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto load_ptr = [=](int u, int i) {" << std::endl;
        return ptr[aux_reg_load_data
            + (i * jcp.os + u) * jcp.oc_block * sizeof(float)];
    };

    auto diff_bias_reg = [=](int i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto diff_bias_reg = [=](int i) {" << std::endl; return Ymm(i); };

    mov(reg_diff_bias_data, ptr[rsp + reg_diff_bias_data_stack_offt]);
    cmp(reg_diff_bias_data, 0);
    je(diff_bias_loop_out, T_NEAR);

    test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
    jz(diff_bias_load, T_NEAR);

    for (int i = 0; i < load_loop_blk; ++i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      for (int i = 0; i < load_loop_blk; ++i) {" << std::endl;
        auto r = diff_bias_reg(i);
        vxorps(r, r, r);
    }
    jmp(diff_bias_init_out, T_NEAR);

    L(diff_bias_load);
    for (int i = 0; i < load_loop_blk; ++i)
        vmovups(diff_bias_reg(i), diff_bias_ptr(i));

    L(diff_bias_init_out);
    mov(aux_reg_load_data, reg_load_data);
    mov(reduce_loop_iter, reg_reduce_loop_work);
    L(diff_bias_loop); {
        for(int u = 0; u < jcp.reduce_loop_unroll; ++u)
            for (int i = 0; i < load_loop_blk; ++i)
                vaddps(diff_bias_reg(i), diff_bias_reg(i), load_ptr(u, i));
        assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jnz(diff_bias_loop, T_NEAR);
    }

    for (int i = 0; i < load_loop_blk; i++)
        vmovups(diff_bias_ptr(i), diff_bias_reg(i));
    add(reg_diff_bias_data, load_loop_blk * jcp.oc_block * sizeof(float));
    mov(ptr[rsp + reg_diff_bias_data_stack_offt], reg_diff_bias_data);

    L(diff_bias_loop_out);
}

void jit_avx2_1x1_conv_kernel_f32::generate()
{
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:  void jit_avx2_1x1_conv_kernel_f32::generate() {" << std::endl;
    const auto &p = attr_.post_ops_;
    int end_idx = jcp.with_dw_conv ? p.find(primitive_kind::convolution) : p.len_;
    for (int i = 0; i < end_idx; i++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      for (int i = 0; i < end_idx; i++) {" << std::endl;
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          if (post_op.is_eltwise()) {" << std::endl;
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<avx2>(
                    this,
                    post_op.eltwise.alg,
                    post_op.eltwise.alpha,
                    post_op.eltwise.beta
            ));
        } else if (post_op.is_depthwise()) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          } else if (post_op.is_depthwise()) {" << std::endl;
        depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<avx2>(
                this,
                post_op.depthwise.alg
        ));
        }
    }

    preamble();

    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);
    if (jcp.with_bias) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      if (jcp.with_bias) {" << std::endl;
        if (jcp.prop_kind == backward_weights) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          if (jcp.prop_kind == backward_weights) {" << std::endl;
            sub(rsp, stack_space_needed);
            mov(reg_diff_bias_data, ptr[param1 + GET_OFF(bias_data)]);
            mov(ptr[rsp + reg_diff_bias_data_stack_offt], reg_diff_bias_data);
        } else
            mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    }

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(first_last_flag)]);
    if (jcp.prop_kind == backward_weights)
        mov(reg_output_stride, ptr[param1 + GET_OFF(output_stride)]);
    mov(reg_oc_off, ptr[param1 + GET_OFF(oc_off)]);

    auto generate_load_loop_body = [=] (int load_loop_blk) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto generate_load_loop_body = [=] (int load_loop_blk) {" << std::endl;
        generate_bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        switch (jcp.prop_kind) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          switch (jcp.prop_kind) {" << std::endl;
        case forward_training:
        case forward_inference:
            add(reg_bias_data, load_loop_blk * jcp.oc_block * sizeof(float));
            if (jcp.with_dw_conv)
                add(reg_output_data,
                    load_loop_blk * jcp.ow * jcp.oc_block * sizeof(float));
            else
                add(reg_output_data,
                    load_loop_blk * jcp.os * jcp.oc_block * sizeof(float));
            break;
        case backward_data:
            add(reg_output_data,
                    load_loop_blk * jcp.is * jcp.ic_block * sizeof(float));
            break;
        case backward_weights:
            for (int i = 0; i < load_loop_blk; i++)
                add(reg_output_data, reg_output_stride);
            break;
        default:
            assert(!"invalid prop_kind");
        }
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
        add(reg_oc_off, load_loop_blk * jcp.oc_block * sizeof(float));
    };

    Label load_loop_blk_8;
    Label load_loop_blk_16;
    Label load_loop_blk_24;
    Label load_loop_blk_end;

    cmp(reg_load_loop_work, 8);
    jle(load_loop_blk_8, T_NEAR);

    cmp(reg_load_loop_work, 32);
    je(load_loop_blk_16, T_NEAR);

    cmp(reg_load_loop_work, 16);
    jle(load_loop_blk_16, T_NEAR);

    L(load_loop_blk_24); {
        generate_diff_bias_loop(3);
        generate_load_loop_body(3);
        cmp(reg_load_loop_work, 32);
        je(load_loop_blk_16);
        cmp(reg_load_loop_work, 24);
        jge(load_loop_blk_24);
    }

    cmp(reg_load_loop_work, 8);
    jle(load_loop_blk_8, T_NEAR);

    L(load_loop_blk_16); {
        generate_diff_bias_loop(2);
        generate_load_loop_body(2);
        cmp(reg_load_loop_work, 16);
        jge(load_loop_blk_16);
    }

    L(load_loop_blk_8); {
        cmp(reg_load_loop_work, 0);
        je(load_loop_blk_end, T_NEAR);
        generate_diff_bias_loop(1);
        generate_load_loop_body(1);
    }

    L(load_loop_blk_end);

    if (jcp.with_bias && jcp.prop_kind == backward_weights)
        add(rsp, 8);

    postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

bool jit_avx2_1x1_conv_kernel_f32::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {" << std::endl;
    const auto &p = attr.post_ops_;

    int dw_conv_idx = p.find(primitive_kind::convolution);
    bool with_dw_conv = dw_conv_idx != -1;

    auto all_post_ops_supported = [&]() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto all_post_ops_supported = [&]() {" << std::endl;
        bool ok = true;

        int end_idx = with_dw_conv ? dw_conv_idx : p.len_;
        for (int i = 0; i < end_idx; i++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          for (int i = 0; i < end_idx; i++) {" << std::endl;
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                    primitive_kind::quantization);
        }
        return ok;
    };
    auto contain = [&](mkldnn::impl::primitive_kind_t kind) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto contain = [&](mkldnn::impl::primitive_kind_t kind) {" << std::endl; return p.find(kind, 0, dw_conv_idx) != -1; };
    auto position = [&](mkldnn::impl::primitive_kind_t kind) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto position = [&](mkldnn::impl::primitive_kind_t kind) {" << std::endl; return p.find(kind, 0, dw_conv_idx); };
    auto count = [&](mkldnn::impl::primitive_kind_t kind) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      auto count = [&](mkldnn::impl::primitive_kind_t kind) {" << std::endl; return p.count(kind, 0, dw_conv_idx); };

    return all_post_ops_supported() &&
           count(primitive_kind::sum) <= 1 &&
           IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == 0) &&
           IMPLICATION(with_dw_conv, !contain(primitive_kind::sum));
}

status_t jit_avx2_1x1_conv_kernel_f32::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr)
{
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          const primitive_attr_t &attr) {" << std::endl;
    if (!mayiuse(avx)) return status::unimplemented;

    // TODO (Roma): this code is duplicated from the generic kernel; maybe the
    // configuration struct could do some stuff below
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][0];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[0];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;

    jcp.src_dt = cd.src_desc.data_type;
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;

    int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;
    jcp.with_dw_conv = dw_conv_ind != -1;
    if (jcp.with_dw_conv) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      if (jcp.with_dw_conv) {" << std::endl;
        jcp.dw_conv_oh = jcp.oh;
        jcp.dw_conv_ow = jcp.ow;
        jcp.oh = p.entry_[dw_conv_ind].dw_conv.in_h;
        jcp.ow = p.entry_[dw_conv_ind].dw_conv.in_w;

        jcp.dw_conv_dst_dt = jcp.dst_dt;
        jcp.dst_dt = p.entry_[dw_conv_ind].dw_conv.in_dt;
    }

    if (jcp.with_dw_conv && !mayiuse(avx2))
        return status::unimplemented;

    if (!mayiuse(avx2)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      if (!mayiuse(avx2)) {" << std::endl;
        for (int i = 0; i < p.len_; i++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          for (int i = 0; i < p.len_; i++) {" << std::endl;
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              if (post_op.is_eltwise()) {" << std::endl;
                if (post_op.eltwise.alg != alg_kind::eltwise_relu)
                    return status::unimplemented;
            } else if (post_op.is_depthwise()) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:              } else if (post_op.is_depthwise()) {" << std::endl;
                return status::unimplemented;
            }
        }
    }

    jcp.with_sum = p.find(primitive_kind::sum, 0, dw_conv_ind) != -1;

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;

    const int is_bwd_d = jcp.prop_kind == backward_data;
    memory_format_t weights_format = with_groups
        ? utils::pick(2 * ndims - 6 + is_bwd_d, gOIw8i8o, gOIw8o8i, gOIhw8i8o,
            gOIhw8o8i)
        : utils::pick(2 * ndims - 6 + is_bwd_d, OIw8i8o, OIw8o8i, OIhw8i8o,
            OIhw8o8i);

    const int simd_w = 8;

    jcp.oc = rnd_up(jcp.oc, simd_w);
    jcp.ic = rnd_up(jcp.ic, simd_w);

    bool args_ok = true
        && jcp.ngroups == 1
        && one_of(src_d.format(), nCw8c, nChw8c)
        && weights_d.format() == weights_format
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && one_of(dst_d.format(), nCw8c, nChw8c);
    if (!args_ok) return status::unimplemented;

    args_ok = true
        && jcp.ih == jcp.oh && jcp.iw == jcp.ow
        && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
        && jcp.t_pad == 0 && jcp.l_pad == 0
        && jcp.stride_w == 1 && jcp.stride_h == 1 // TODO: support some strides
        && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    // TODO: remove this restriction
    // optimized 1x1 bwd_w does not support Intel AVX
    if (jcp.prop_kind == backward_weights && !mayiuse(avx2))
        return status::unimplemented;

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.ur = mayiuse(avx2) ? 4 : 3; // Intel AVX support

    int load_blocking{ 0 };
    int load_blocking_max{ 0 };
    int bcast_blocking{ 0 };
    int bcast_blocking_max{ 0 };
    int reduce_blocking{ 0 };

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      if (one_of(jcp.prop_kind, forward_training, forward_inference)) {" << std::endl;
        jcp.reduce_dim = jcp.ic;
        jcp.reduce_block = jcp.ic_block;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.with_dw_conv ? jcp.iw : jcp.is;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.is * sizeof(float);
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.oc_block * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur * jcp.oc_block * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.ic * jcp.oc_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        load_blocking = jcp.with_dw_conv ? nstl::min(3 * jcp.load_block, jcp.oc) : 120; // assumes the kernel is jcp.ur x 3
        load_blocking_max = jcp.with_dw_conv ? nstl::min(3 * jcp.load_block, jcp.oc) : 144;
        bcast_blocking = 128; // affects load balancing across threads
        bcast_blocking_max = 192;
        reduce_blocking = 128; // affects L1$ utilization
    } else if (jcp.prop_kind == backward_data) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      } else if (jcp.prop_kind == backward_data) {" << std::endl;
        jcp.reduce_dim = jcp.oc;
        jcp.reduce_block = jcp.oc_block;

        jcp.load_dim = jcp.ic;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.os;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.os * sizeof(float);
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.ic * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.oc_block * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.ic_block;

        load_blocking = 96; // assumes the kernel is jcp.ur x 3
        load_blocking_max = 144;
        bcast_blocking = 128; // affects load balancing across threads
        bcast_blocking_max = 196;
        reduce_blocking = 64; // affects L1$ utilization
    } else if (jcp.prop_kind == backward_weights) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      } else if (jcp.prop_kind == backward_weights) {" << std::endl;
        jcp.reduce_dim = jcp.os;
        jcp.reduce_block = 1;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.ic_block * sizeof(float);
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.oc_block * sizeof(float);

        jcp.bcast_loop_output_step = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = jcp.oc_block * jcp.ur * sizeof(float);
        jcp.bcast_loop_bcast_step = jcp.ic_block * jcp.is * sizeof(float);
        jcp.bcast_loop_bcast_substep = jcp.ur * sizeof(float);

        jcp.load_loop_load_step = jcp.oc_block * jcp.os * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        while (true) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          while (true) {" << std::endl;
            if (load_blocking <= 32) break;
            else if (load_blocking % 2 == 0) load_blocking /= 2;
            else if (load_blocking % 3 == 0) load_blocking /= 3;
            else break;
        }
        load_blocking *= jcp.load_block;
        load_blocking_max = load_blocking;
        assert(jcp.load_dim % load_blocking == 0);

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        while (true) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          while (true) {" << std::endl;
            if (bcast_blocking <= 9) break;
            else if (bcast_blocking % 2 == 0) bcast_blocking /= 2;
            else if (bcast_blocking % 3 == 0) bcast_blocking /= 3;
            else break;
        }
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(jcp.bcast_dim % bcast_blocking == 0);

        reduce_blocking = 128; // affects L1$ utilization
    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);

    assert(jcp.bcast_block % jcp.ur == 0);
    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;

    jcp.nb_bcast = jcp.with_dw_conv ? jcp.ih : div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    return status::success;
}

void jit_avx2_1x1_conv_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp, const jit_conv_conf_t &jcp_dw) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:          const jit_1x1_conv_conf_t &jcp, const jit_conv_conf_t &jcp_dw) {" << std::endl;
    using namespace mkldnn::impl::memory_tracking::names;

    if (jcp.prop_kind != backward_data && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, sizeof(float) * jcp.oc);

    if (jcp.with_dw_conv) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_avx2_1x1_conv_kernel_f32.cpp:      if (jcp.with_dw_conv) {" << std::endl;
        const int nthreads = mkldnn_get_max_threads();
        size_t dw_conv_buffer_size_ = (size_t)jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block * (jcp.oc / jcp.oc_block);
        scratchpad.book(key_dw_conv_buffer, sizeof(float) * dw_conv_buffer_size_ * nthreads);

        if (jcp.oc != jcp.oc_without_padding)
            scratchpad.book(key_dw_conv_padded_bias, sizeof(float) * jcp.oc);
    }
}

}
}
}
