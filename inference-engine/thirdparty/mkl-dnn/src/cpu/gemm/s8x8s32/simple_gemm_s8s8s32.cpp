#include <iostream>
/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#include <cstdint>

#include "simple_gemm_s8s8s32.hpp"

#include "jit_generator.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_types.h"
#include "nstl.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

void compensation_init(const char *offsetC, int32_t *compensation, int len,
        const int32_t *oc) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:          const int32_t *oc) {" << std::endl;
    bool OCisC = (*offsetC == 'C' || *offsetC == 'c');
    bool OCisF = (*offsetC == 'F' || *offsetC == 'f');

   if (OCisF && (*oc) != 0) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:     if (OCisF && (*oc) != 0) {" << std::endl;
       for (int i = 0; i < len; i++)
           compensation[i] = *oc;
   } else if (OCisC) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:     } else if (OCisC) {" << std::endl;
       for (int i = 0; i < len; i++)
           compensation[i] = oc[i];
   } else {
       for (int i = 0; i < len; i++)
           compensation[i] = 0;
   }
}

void compensation_compute(bool transa, int m, int k, float alpha,
        const int8_t *a, int lda, int32_t *compensation) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:          const int8_t *a, int lda, int32_t *compensation) {" << std::endl;
    if (!transa) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:      if (!transa) {" << std::endl;
        const int L2_cache_size = get_cache_size(2, true);
        const int blocking_factor = nstl::min(k, L2_cache_size / lda + 1);
        const int npanels = k / blocking_factor;
        const bool has_tile = k % blocking_factor > 0;

        parallel_nd(npanels, m, [&](int j, int i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:          parallel_nd(npanels, m, [&](int j, int i) {" << std::endl;
            int32_t val = 0;
            for (int jb = 0; jb < blocking_factor; jb++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:              for (int jb = 0; jb < blocking_factor; jb++) {" << std::endl;
                val += a[(i + (ptrdiff_t)j * blocking_factor * lda)
                    + (ptrdiff_t)jb * lda];
            }
            if (alpha != 1.0f) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:              if (alpha != 1.0f) {" << std::endl;
                val = math::out_round<int32_t>(math::saturate<int32_t>(
                    (double)val * alpha * -128.0));
            } else {
                val *= -128;
            }
            mkldnn_fetch_and_add(&compensation[i], val);
        });

        if (has_tile) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:          if (has_tile) {" << std::endl;
            parallel_nd(m, [=](int i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:              parallel_nd(m, [=](int i) {" << std::endl;
                int32_t val = 0;
                for (int j = npanels * blocking_factor; j < k; j++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:                  for (int j = npanels * blocking_factor; j < k; j++) {" << std::endl;
                    val += a[i + (ptrdiff_t)j * lda];
                }
                if (alpha != 1.0f) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:                  if (alpha != 1.0f) {" << std::endl;
                    val = math::out_round<int32_t>(math::saturate<int32_t>(
                        (double)val * alpha * -128.0));
                } else {
                    val *= -128;
                }
                mkldnn_fetch_and_add(&compensation[i], val);
            });
        }
    } else {
        parallel_nd(m, [=](int i) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:          parallel_nd(m, [=](int i) {" << std::endl;
            int32_t val = 0;
            for (int j = 0; j < k; j++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:              for (int j = 0; j < k; j++) {" << std::endl;
                val += a[j + (ptrdiff_t)i * lda];
            }
            if (alpha != 1.0f) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:              if (alpha != 1.0f) {" << std::endl;
                val = math::out_round<int32_t>(math::saturate<int32_t>(
                    (double)val * alpha * -128.0));
            } else {
                val *= -128;
            }
            compensation[i] += val;
        });
    }
}

void copy_and_shift_b(bool transb, int k, int n, uint8_t *b_u8, int ldb_u8,
        const int8_t *b_s8, int ldb_s8) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:          const int8_t *b_s8, int ldb_s8) {" << std::endl;
    const int b_cols = transb ? k : n;

    parallel_nd(b_cols, [=](int j) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:      parallel_nd(b_cols, [=](int j) {" << std::endl;
        const int b_rows = transb ? n : k;

        uint8_t *pb_u8 = b_u8 + j * ldb_u8;
        const int8_t *pb_s8 = b_s8 + j * ldb_s8;

        for (int i = 0; i < b_rows; i++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:          for (int i = 0; i < b_rows; i++) {" << std::endl;
            (*pb_u8) = (*pb_s8) + 128;
            pb_u8++;
            pb_s8++;
        }
    });
}

/**
 * gemm_s8s8s32 operation is defined as follows:
 * C = alpha * op(A) * (op(B) + B_shift) + beta * C + C_offset + compensation
 *
 * where
 *  - compensation is a vector of length m that contains computed compensation
 *   that may contain C_offset if applicable. The compensation is applied inside
 *   gemm_s8u8s32 as a C_offset
 *  - B_shift is a k-by-n matrix, every element of B_shift is equal to 128
 *
 *  What is the compensation:
 *  In order to prepare the matrix B for gemm_s8u8s32 call the B_shift is applied:
 *  C = alpha * op(A) * (op(B) + B_shift) + beta * C + C_offset =
 *  alpha * op(A) * op(B) + alpha * op(A) * B_shift + beta * C + C_offset
 *  compensation = -alpha * op(A) * B_shift
 *  Since B_shift is a matrix, every element of which is equal to 128 then
 *  - if op(A) = A: compensation contains sum of the elements in each row
 *   scaled by -128 * alpha
 *  - if op(A) = A**T: compensation contains sum of the elements in each column
 *   scaled by -128 * alpha
 *
 * The rest of parameters is described in mkldnn.h
 */
mkldnn_status_t simple_gemm_s8s8s32(
        const char *transA, const char *transB, const char *offsetC,
        const int *m, const int *n, const int *k,
        const float *alpha, const int8_t *a, const int *lda, const int8_t *oa,
        const int8_t *b, const int *ldb, const int8_t *ob,
        const float *beta, int32_t *c, const int *ldc, const int32_t *oc) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:          const float *beta, int32_t *c, const int *ldc, const int32_t *oc) {" << std::endl;
    if (*oa != 0 || *ob != 0) return mkldnn_unimplemented;

    int M = *m, N = *n, K = *k;
    bool transa = (*transA == 'T' || *transA == 't');
    bool transb = (*transB == 'T' || *transB == 't');
    int ld = transb ? N : K;

    uint8_t *b_u8 = (uint8_t *)malloc(sizeof(uint8_t) * K * N, 64);
    int32_t *compensation = (int32_t *)malloc(sizeof(int32_t) * M, 64);

    if (utils::any_null(b_u8, compensation)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:      if (utils::any_null(b_u8, compensation)) {" << std::endl;
        free(b_u8);
        free(compensation);
        return mkldnn_out_of_memory;
    }

    compensation_init(offsetC, compensation, M, oc);
    compensation_compute(transa, M, K, *alpha, a, *lda, compensation);
    copy_and_shift_b(transb, K, N, b_u8, ld, b, *ldb);

    mkldnn_gemm_s8u8s32(transA, transB, "C", m, n, k, alpha, a, lda, oa, b_u8,
        &ld, ob, beta, c, ldc, compensation);

    if ((*offsetC == 'R' || *offsetC == 'r'))
        parallel_nd(M, N,
            [=](int i, int j) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp:              [=](int i, int j) {" << std::endl; c[i + (ptrdiff_t)j * *ldc] += oc[j]; });

    free(b_u8);
    free(compensation);

    return mkldnn_success;
}
}
}
}
