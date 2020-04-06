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

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "mkldnn_debug.h"
#include "nstl.hpp"
#include "type_helpers.hpp"

#include "cpu_primitive.hpp"
#include "cpu_reorder_pd.hpp"
#include "jit_uni_reorder.hpp"

#include "jit_avx512_core_bf16cvt.hpp"
#include "jit_generator.hpp"

// #define TR_DEBUG
#if defined(TR_DEBUG)
#define DEBUg(...) do { __VA_ARGS__ } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

#ifdef _WIN32
/* seems like s_addr is a reserved macro on Windows */
#undef s_addr
#endif

using namespace Xbyak;
using namespace mkldnn::impl::types;

namespace mkldnn {
namespace impl {
namespace cpu {

namespace tr {

/** Minimal reasonable/desirable kernel size.
 * The constant might be used to determine how a problem should be split
 * between kernel and threading driver. */
const size_t ker_prb_size_min = 64;

/* kernel */
struct jit_uni_reorder_kernel_f32: public kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reorder_kernel_f32)

    enum {
        len_unroll_max = 256,
        ndims_jit_loop_max = 3,
    };

    struct simple_impl_desc_t {
        int ndims_full_unroll;
        int len_last_dim_unroll;
        int len_unroll;
    };

    static bool simple_impl_desc_init(const prb_t &prb,
            simple_impl_desc_t *desc) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              simple_impl_desc_t *desc) {" << std::endl;
        const int ndims = prb.ndims;

        int ndims_full_unroll = 0;
        int len_last_dim_unroll = 1;
        int len_unroll = 1;

        for (int d = 0; d < ndims; ++d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int d = 0; d < ndims; ++d) {" << std::endl;
            auto &node = prb.nodes[d];
            if (len_unroll * node.n <= len_unroll_max) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (len_unroll * node.n <= len_unroll_max) {" << std::endl;
                ndims_full_unroll++;
                len_unroll *= node.n;
            } else {
                len_last_dim_unroll = len_unroll_max / len_unroll;
                while (node.n % len_last_dim_unroll)
                    --len_last_dim_unroll;
                len_unroll *= len_last_dim_unroll;
                break;
            }
        }

        if (prb.ndims - ndims_full_unroll > ndims_jit_loop_max)
            return false;

        if (desc) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          if (desc) {" << std::endl;
            desc->ndims_full_unroll = ndims_full_unroll;
            desc->len_last_dim_unroll = len_last_dim_unroll;
            desc->len_unroll = len_unroll;
        }

        return true;
    }

    static bool applicable(const prb_t &p) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      static bool applicable(const prb_t &p) {" << std::endl;
        using namespace data_type;

        bool ok = true
            && p.ndims > 0
            && utils::one_of(p.itype, f32, bf16, s32, s8, u8)
            && utils::one_of(p.otype, f32, bf16, s32, s8, u8)
            && IMPLICATION(p.itype == bf16, utils::one_of(p.otype, f32, bf16))
            && IMPLICATION(p.otype == bf16, utils::one_of(p.itype, f32, bf16))
            && utils::everyone_is(0, p.ioff, p.ooff) /* do we need this? */
            && utils::one_of(p.beta, 0.f, 1.f) /* anything else? */
            && simple_impl_desc_init(p, nullptr)
            && mayiuse(sse42)
            && IMPLICATION(!utils::everyone_is(f32, p.itype, p.otype),
                    mayiuse(avx))
            && IMPLICATION((p.itype == bf16 || p.otype == bf16),
                    mayiuse(avx512_core));
        if (!ok) return false;

        const ptrdiff_t max_stride = (1LL<<31) - 1;
        for (int d = 0; d < p.ndims; ++d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int d = 0; d < p.ndims; ++d) {" << std::endl;
            const ptrdiff_t cms = max_stride / p.nodes[d].n;
            bool strides_ok = true
                && p.nodes[d].is < cms / (int)data_type_size(p.itype)
                && p.nodes[d].os < cms / (int)data_type_size(p.otype);
            if (!strides_ok) return false;
        }

        return true;
    }

    int n(int d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      int n(int d) {" << std::endl; assert(d < prb_.ndims); return (int)prb_.nodes[d].n; }
    int is(int d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      int is(int d) {" << std::endl; assert(d < prb_.ndims); return (int)prb_.nodes[d].is; }
    int os(int d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      int os(int d) {" << std::endl; assert(d < prb_.ndims); return (int)prb_.nodes[d].os; }
    int ss(int d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      int ss(int d) {" << std::endl; assert(d < prb_.ndims); return (int)prb_.nodes[d].ss; }

    Address i_addr(int i_off)
    {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      Address i_addr(int i_off)     {" << std::endl; return ptr[reg_ptr_in + reg_off_in + i_off * itype_sz]; }

    Address o_addr(int o_off)
    {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      Address o_addr(int o_off)     {" << std::endl; return ptr[reg_ptr_out + reg_off_out + o_off * otype_sz]; }

    Address s_addr(int s_off)
    {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      Address s_addr(int s_off)     {" << std::endl; return ptr[reg_ptr_scale + reg_off_scale + s_off * stype_sz]; }

    void step(int off, int prev_i_off, int prev_o_off, int prev_s_off,
            int &i_off, int &o_off, int &s_off, int step_size = 1) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              int &i_off, int &o_off, int &s_off, int step_size = 1) {" << std::endl;
        i_off = prev_i_off;
        o_off = prev_o_off;
        s_off = prev_s_off;

        if (off == 0) return;

        int start_dim = 0, dims_prod = 1;
        for (; start_dim < prb_.ndims && dims_prod != step_size; ++start_dim)
            dims_prod *= n(start_dim);
        assert(start_dim < prb_.ndims);
        off /= step_size;

        for (int d = start_dim; d < prb_.ndims; ++d) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int d = start_dim; d < prb_.ndims; ++d) {" << std::endl;
            i_off += is(d);
            o_off += os(d);
            s_off += ss(d);

            if (off % n(d)) break;

            i_off += - n(d) * is(d);
            o_off += - n(d) * os(d);
            s_off += - n(d) * ss(d);
            off /= n(d);

            if (off == 0) break; /* FIXME: is it really required? */
        }
    }

    void step(int off, int prev_i_off, int prev_o_off, int &i_off, int &o_off,
            int step_size = 1) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              int step_size = 1) {" << std::endl;
        int dummy = 0;
        step(off, prev_i_off, prev_o_off, dummy, i_off, o_off, dummy,
                step_size);
    }

    void tr8x8_avx2(int i_off, int o_off) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      void tr8x8_avx2(int i_off, int o_off) {" << std::endl;
        for (int i = 0; i < 8; i++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int i = 0; i < 8; i++) {" << std::endl;
            using namespace data_type;

            if (prb_.itype == s32 && prb_.otype == f32)
                vcvtdq2ps(Ymm(i), i_addr(i_off + i * 8));
            else if (prb_.itype == f32 && prb_.otype == s32)
                vcvtps2dq(Ymm(i), i_addr(i_off + i * 8));
            else
                vmovups(Ymm(i), i_addr(i_off + i * 8));
        }

        for (int i = 0; i < 8 / 2; i++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int i = 0; i < 8 / 2; i++) {" << std::endl;
            vunpcklps(Ymm(8 + i), Ymm(2 * i), Ymm(2 * i + 1));
            vunpckhps(Ymm(i), Ymm(2 * i), Ymm(2 * i + 1));
        }

        const unsigned int lfloat = 0x44;
        const unsigned int ufloat = 0xee;
        for (int i = 0; i < 8 / 2; i++) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int i = 0; i < 8 / 2; i++) {" << std::endl;
            int j = i % 2 == 0 ? 8 + i : i - 1;
            vshufps(Ymm(8 / 2 + 2 * i), Ymm(j), Ymm(j + 1), lfloat);
            vshufps(Ymm(8 / 2 + 2 * i + 1), Ymm(j), Ymm(j + 1), ufloat);
        }

        const unsigned int lquad = 0x20;
        for (int i = 0; i < 8 / 2; i++)
            vperm2f128(Ymm(i), Ymm(8 / 2 + i), Ymm(8 + i), lquad);

        const unsigned int uquad = 0x31;
        for (int i = 8 / 2; i < 8; i++)
            vperm2f128(Ymm(i), Ymm(i), Ymm(8 / 2 + i), uquad);

        for (int i = 0; i < 8; i++)
            vmovups(o_addr(o_off + i * 8), Ymm(i));
    }

    bool process_unroll_tr8x8(int len) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      bool process_unroll_tr8x8(int len) {" << std::endl;
        bool can_do = true
            && mayiuse(avx2)
            && prb_.ndims >= 2
            && utils::everyone_is(4, itype_sz, otype_sz)
            && utils::everyone_is(8, n(0), n(1))
            && utils::everyone_is(1, os(0), is(1))
            && utils::everyone_is(8, os(1), is(0))
            && prb_.scale_type == scale_type_t::NONE
            && prb_.beta == 0.f;
        if (!can_do) return false;

        const int step_size = n(0) * n(1);
        int i_off = 0, o_off = 0;
        for (int off = 0; off < len; off += step_size) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int off = 0; off < len; off += step_size) {" << std::endl;
            step(off, i_off, o_off, i_off, o_off, step_size);
            tr8x8_avx2(i_off, o_off);
        }

        return true;
    }

    template <cpu_isa_t isa>
    bool process_direct_copy(int len) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      bool process_direct_copy(int len) {" << std::endl;
        using namespace data_type;

        using Vmm = typename cpu_isa_traits<isa>::Vmm;
        const int simd_w = cpu_isa_traits<isa>::vlen / itype_sz;

        bool can_do = true
            && mayiuse(isa)
            && utils::everyone_is(1, os(0), is(0))
            && (false
                    || prb_.itype == prb_.otype
                    || (prb_.itype == s32 && prb_.otype == f32)
                    || (prb_.itype == f32 && prb_.otype == s32)
                    )
            && len % simd_w == 0
            && n(0) % len == 0
            && prb_.scale_type == scale_type_t::NONE
            && prb_.beta == 0.f;
        if (!can_do) return false;

        for (int off = 0; off < len;) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int off = 0; off < len;) {" << std::endl;
            const int unroll = nstl::min(16, (len - off) / simd_w);

            for (int ur = 0; ur < unroll; ++ur)
                uni_vmovups(Vmm(ur), i_addr(off + ur * simd_w));

            if (prb_.itype != prb_.otype) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (prb_.itype != prb_.otype) {" << std::endl;
                for (int ur = 0; ur < unroll; ++ur) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  for (int ur = 0; ur < unroll; ++ur) {" << std::endl;
                    if (prb_.itype == s32 && prb_.otype == f32)
                        uni_vcvtdq2ps(Vmm(ur), Vmm(ur));
                    else if (prb_.itype == f32 && prb_.otype == s32)
                        uni_vcvtps2dq(Vmm(ur), Vmm(ur));
                    else assert(!"unreachable");
                }
            }

            for (int ur = 0; ur < unroll; ++ur)
                uni_vmovups(o_addr(off + ur * simd_w), Vmm(ur));

            off += unroll * simd_w;
        }

        return true;
    }

    void process_unroll_generic_step(int reg_unroll, const int *i_off,
            const int *o_off, const int *s_off) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              const int *o_off, const int *s_off) {" << std::endl;
        using namespace data_type;

        auto cvt2ps = [=](const Xmm &dst, const Operand &src, data_type_t idt) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          auto cvt2ps = [=](const Xmm &dst, const Operand &src, data_type_t idt) {" << std::endl;
            Xmm dst_pure = Xmm(dst.getIdx());
            switch (idt) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              switch (idt) {" << std::endl;
            case f32:
                if (src.isMEM() || src.getIdx() != dst.getIdx())
                    vmovups(dst, src);
                break;
            case bf16: vpmovzxwd(dst, src); vpslld(dst, dst, 0x10); break;
            case s32: vcvtdq2ps(dst, src); break;
            case s8: vpmovsxbd(dst, src); vcvtdq2ps(dst_pure, dst); break;
            case u8: vpmovzxbd(dst, src); vcvtdq2ps(dst_pure, dst); break;
            default: assert(!"unreachable");
            }
        };

        auto cvt2odt = [=](const Xmm &xmm, data_type_t odt, data_type_t idt) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          auto cvt2odt = [=](const Xmm &xmm, data_type_t odt, data_type_t idt) {" << std::endl;
            switch (odt) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              switch (odt) {" << std::endl;
            case bf16:
                if (idt == f32) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  if (idt == f32) {" << std::endl;
                    if (is_cpx_) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                      if (is_cpx_) {" << std::endl;
                        vcvtneps2bf16(xmm, xmm);
                    } else {
                        bf16_emu_->r_vcvtneps2bf16(
                                Ymm(xmm.getIdx()), Zmm(xmm.getIdx()));
                    }
                }
                break;
            case s32:
                if (idt == f32) vcvtps2dq(xmm, xmm);
                else if (idt == s8) vpmovsxbd(xmm, xmm);
                else if (idt == u8) vpmovzxbd(xmm, xmm);
                break;
            case s8:
                if (idt == f32) vcvtps2dq(xmm, xmm);
                if (idt == f32 || idt == s32) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  if (idt == f32 || idt == s32) {" << std::endl;
                    if (mayiuse(avx512_core)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                      if (mayiuse(avx512_core)) {" << std::endl;
                        vpmovsdb(xmm, xmm);
                    } else {
                        vpackssdw(xmm, xmm, xmm_zero);
                        vpacksswb(xmm, xmm, xmm_zero);
                    }
                }
                if (idt == u8) vpminub(xmm, xmm, xmm_4x127b);
                break;
            case u8:
                if (idt == f32) vcvtps2dq(xmm, xmm);
                if (idt == f32 || idt == s32) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  if (idt == f32 || idt == s32) {" << std::endl;
                    if (mayiuse(avx512_core)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                      if (mayiuse(avx512_core)) {" << std::endl;
                        vpmaxsd(xmm, xmm, xmm_zero);
                        vpmovusdb(xmm, xmm);
                    } else {
                        vpackssdw(xmm, xmm, xmm_zero);
                        vpackuswb(xmm, xmm, xmm_zero);
                    }
                }
                if (idt == s8) vpmaxsb(xmm, xmm, xmm_zero);
                break;
            default: assert(!"unreachable");
            }
        };

        auto load = [=](const Xmm &xmm, const Address &addr, int size) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          auto load = [=](const Xmm &xmm, const Address &addr, int size) {" << std::endl;
            switch (size) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              switch (size) {" << std::endl;
            case 16: movups(xmm, addr); break;
            case 8: movsd(xmm, addr); break;
            case 4: movss(xmm, addr); break;
            case 2: pinsrw(xmm, addr, 0x0); break;
            case 1: pinsrb(xmm, addr, 0x0); break;
            default: assert(!"unreachable");
            }
        };

        auto store = [=](const Address &addr, const Xmm &xmm, int size) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          auto store = [=](const Address &addr, const Xmm &xmm, int size) {" << std::endl;
            switch (size) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              switch (size) {" << std::endl;
            case 16: movups(addr, xmm); break;
            case 8: movsd(addr, xmm); break;
            case 4: movss(addr, xmm); break;
            case 2: pextrw(addr, xmm, 0x0); break;
            case 1: pextrb(addr, xmm, 0x0); break;
            default: assert(!"unreachable");
            }
        };

        /* check whether loading 4 values at once is possible */
        bool can_load_xmm = mayiuse(avx) && reg_unroll % 4 == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (i_off[ur] != i_off[ur - 1] + 1)
                can_load_xmm = false;
        const int load_step = can_load_xmm ? 4 : 1;

        /* check whether storing 4 values at once is possible */
        bool can_store_xmm = reg_unroll % 4 == 0;
        for (int ur = 1; ur < reg_unroll; ++ur)
            if (o_off[ur] != o_off[ur - 1] + 1)
                can_store_xmm = false;
        const int ur_step = can_store_xmm ? 4 : 1;

        const bool interim_f32 = false
            || utils::one_of(f32, prb_.itype, prb_.otype)
            || prb_.scale_type != scale_type_t::NONE
            || prb_.beta != 0.f;

        if (!can_load_xmm && can_store_xmm) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          if (!can_load_xmm && can_store_xmm) {" << std::endl;
            assert(ur_step == 4);
            /* load with stride */
            for (int ur = 0; ur < reg_unroll; ur += ur_step) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              for (int ur = 0; ur < reg_unroll; ur += ur_step) {" << std::endl;
                for (int r = 0; r < ur_step; ++r) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  for (int r = 0; r < ur_step; ++r) {" << std::endl;
                    if (itype_sz == 4)
                        pinsrd(Xmm(ur), i_addr(i_off[ur + r]), r);
                    else if (itype_sz == 2)
                        pinsrw(Xmm(ur), i_addr(i_off[ur + r]), r);
                    else
                        pinsrb(Xmm(ur), i_addr(i_off[ur + r]), r);
                }
            }
        } else {
            for (int ur = 0; ur < reg_unroll; ur += load_step)
                load(Xmm(ur), i_addr(i_off[ur]), load_step * itype_sz);
        }

        /* xmm[:] <-- (f32)xmm[:] */
        if (interim_f32) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          if (interim_f32) {" << std::endl;
            const int cvt_step = nstl::max(load_step, ur_step);
            for (int ur = 0; ur < reg_unroll; ur += cvt_step)
                cvt2ps(Xmm(ur), Xmm(ur), prb_.itype);
        }

        if (can_load_xmm && !can_store_xmm) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          if (can_load_xmm && !can_store_xmm) {" << std::endl;
            const bool fast_return = true // transposition on the fly
                && prb_.scale_type != scale_type_t::MANY
                && prb_.beta == 0.f;
            if (fast_return) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (fast_return) {" << std::endl;
                for (int ur = 0; ur < reg_unroll; ur += load_step) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  for (int ur = 0; ur < reg_unroll; ur += load_step) {" << std::endl;
                    if (prb_.scale_type == scale_type_t::COMMON)
                        mulps(Xmm(ur), xmm_scale);
                    if (prb_.otype != f32)
                        cvt2odt(Xmm(ur), prb_.otype,
                                interim_f32 ? f32 : prb_.itype);
                    for (int r = 0; r < load_step; ++r) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                      for (int r = 0; r < load_step; ++r) {" << std::endl;
                        if (otype_sz == 4)
                            pextrd(o_addr(o_off[ur + r]), Xmm(ur), r);
                        else if (otype_sz == 2)
                            pextrw(o_addr(o_off[ur + r]), Xmm(ur), r);
                        else
                            pextrb(o_addr(o_off[ur + r]), Xmm(ur), r);
                    }
                }
                return;
            }

            /* scatter elements of xmm into 4 xmms */
            if (itype_sz == 4 || interim_f32) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (itype_sz == 4 || interim_f32) {" << std::endl;
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r)
                        vshufps(Xmm(ur + r), Xmm(ur), Xmm(ur), r);
            } else if (itype_sz == 2) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              } else if (itype_sz == 2) {" << std::endl;
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r)
                        vpalignr(Xmm(ur + r), Xmm(ur), Xmm(ur), 2 * r);
            } else {
                for (int ur = 0; ur < reg_unroll; ur += load_step)
                    for (int r = 1; r < load_step; ++r)
                        vpalignr(Xmm(ur + r), Xmm(ur), Xmm(ur), r);
            }
        }

        /* scale and beta processing */
        if (can_store_xmm) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          if (can_store_xmm) {" << std::endl;
            /* xmm <-- scale * xmm[:] */
            if (prb_.scale_type == scale_type_t::COMMON) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (prb_.scale_type == scale_type_t::COMMON) {" << std::endl;
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    mulps(Xmm(ur), xmm_scale);
            } else if (prb_.scale_type == scale_type_t::MANY) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              } else if (prb_.scale_type == scale_type_t::MANY) {" << std::endl;
                enum class scale_load_type_t { bcast, load, gather };

                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  for (int ur = 0; ur < reg_unroll; ur += ur_step) {" << std::endl;
                    scale_load_type_t scale_load_type =
                        scale_load_type_t::bcast; // the best case

                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 0)
                            scale_load_type = scale_load_type_t::load;

                    if (scale_load_type == scale_load_type_t::bcast) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                      if (scale_load_type == scale_load_type_t::bcast) {" << std::endl;
                        movss(xmm_scale, s_addr(s_off[ur]));
                        shufps(xmm_scale, xmm_scale, 0x0);
                        mulps(Xmm(ur), xmm_scale);
                        continue;
                    }

                    // bcast doesn't work, the next try -- load
                    for (int r = ur + 1; r < ur + ur_step; ++r)
                        if (s_off[r] != s_off[r - 1] + 1)
                            scale_load_type = scale_load_type_t::gather;

                    if (scale_load_type == scale_load_type_t::load) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                      if (scale_load_type == scale_load_type_t::load) {" << std::endl;
                        movups(xmm_scale, s_addr(s_off[ur]));
                        mulps(Xmm(ur), xmm_scale);
                        continue;
                    }

                    // load doesn't work as well
                    // so gather the scale factors one by one
                    for (int r = ur; r < ur + ur_step; ++r)
                        pinsrd(xmm_scale, s_addr(s_off[r]), r - ur);
                    mulps(Xmm(ur), xmm_scale);
                }
            }

            /* dst <-- beta * dst + xmm[:] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (prb_.beta == 1.f) {" << std::endl;
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  for (int ur = 0; ur < reg_unroll; ur += ur_step) {" << std::endl;
                    if (prb_.otype == f32) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                      if (prb_.otype == f32) {" << std::endl;
                        /* non VEX instructions do not support unaligned
                         * memory for instructions other than movups. */
                        if (mayiuse(avx)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                          if (mayiuse(avx)) {" << std::endl;
                            vaddps(Xmm(ur), o_addr(o_off[ur]));
                        } else {
                            /* register xmm(1) is unused */
                            movups(Xmm(1), o_addr(o_off[ur]));
                            addps(Xmm(ur), Xmm(1));
                        }
                    } else {
                        cvt2ps(Xmm(1), o_addr(o_off[ur]), prb_.otype);
                        vaddps(Xmm(ur), Xmm(1));
                    }
                }
            }
        } else {
            /* xmm[0] <-- scale * xmm[0] */
            if (prb_.scale_type == scale_type_t::COMMON) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (prb_.scale_type == scale_type_t::COMMON) {" << std::endl;
                for (int ur = 0; ur < reg_unroll; ur += ur_step)
                    mulss(Xmm(ur), xmm_scale);
            } else if (prb_.scale_type == scale_type_t::MANY) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              } else if (prb_.scale_type == scale_type_t::MANY) {" << std::endl;
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  for (int ur = 0; ur < reg_unroll; ur += ur_step) {" << std::endl;
                    mulss(Xmm(ur), s_addr(s_off[ur]));
                }
            }

            /* dst <-- beta * dst + xmm[0] */
            assert(prb_.beta == 0.f || prb_.beta == 1.f);
            if (prb_.beta == 1.f) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (prb_.beta == 1.f) {" << std::endl;
                for (int ur = 0; ur < reg_unroll; ur += ur_step) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  for (int ur = 0; ur < reg_unroll; ur += ur_step) {" << std::endl;
                    if (prb_.otype == f32) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                      if (prb_.otype == f32) {" << std::endl;
                        addss(Xmm(ur), o_addr(o_off[ur]));
                    } else {
                        if (prb_.otype == s32) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                          if (prb_.otype == s32) {" << std::endl;
                            vmovss(xmm_tmp, o_addr(o_off[ur]));
                        } else if (utils::one_of(prb_.otype, s8, u8)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                          } else if (utils::one_of(prb_.otype, s8, u8)) {" << std::endl;
                            pinsrb(xmm_tmp, o_addr(o_off[ur]), 0x0);
                        } else {
                            assert(!"unsupported o_type");
                        }
                        cvt2ps(xmm_tmp, xmm_tmp, prb_.otype);
                        addps(Xmm(ur), xmm_tmp);
                    }
                }
            }
        }

        for (int ur = 0; ur < reg_unroll; ur += ur_step) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int ur = 0; ur < reg_unroll; ur += ur_step) {" << std::endl;
            if (prb_.otype != f32)
                cvt2odt(Xmm(ur), prb_.otype, interim_f32 ? f32 : prb_.itype);
            store(o_addr(o_off[ur]), Xmm(ur), ur_step * otype_sz);
        }
    }

    void process_unroll_generic(int len) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      void process_unroll_generic(int len) {" << std::endl;
        const int blk = 8;

        int i_off[2 * blk] = {0};
        int o_off[2 * blk] = {0};
        int s_off[2 * blk] = {0};

        int curr = 0; // will switch between 0 and 1

        for (int off = 0; off < len; off += blk) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for (int off = 0; off < len; off += blk) {" << std::endl;
            const int reg_unroll = nstl::min(off + blk, len) - off;

            /* compute offsets */
            for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              for (int ur = off != 0 ? 0 : 1; ur < reg_unroll; ++ur) {" << std::endl;
                const int ur_c = curr * blk + ur;
                const int ur_p = (ur_c - 1 + 2 * blk) % (2 * blk); // prev ur
                step(off + ur,
                        i_off[ur_p], o_off[ur_p], s_off[ur_p],
                        i_off[ur_c], o_off[ur_c], s_off[ur_c]);
            }

            process_unroll_generic_step(reg_unroll, i_off + curr * blk,
                    o_off + curr * blk, s_off + curr * blk);

            curr = 1 - curr;
        }
    }

    void loop_begin(Label &l, Reg64 reg_cnt, int len) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      void loop_begin(Label &l, Reg64 reg_cnt, int len) {" << std::endl;
        mov(reg_cnt, len);
        L(l);
    }

    void loop_end(Label &l, Reg64 reg_cnt, int len,
            int i_step, int o_step, int s_step) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              int i_step, int o_step, int s_step) {" << std::endl;
        add(reg_off_in, i_step * itype_sz);
        add(reg_off_out, o_step * otype_sz);
        if (prb_.scale_type == scale_type_t::MANY)
            add(reg_off_scale, s_step * stype_sz);
        dec(reg_cnt);
        jnz(l);

        sub(reg_off_in, len * i_step * itype_sz);
        sub(reg_off_out, len * o_step * otype_sz);
        if (prb_.scale_type == scale_type_t::MANY)
            sub(reg_off_scale, len * s_step * stype_sz);
    }

    bool simple_impl() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      bool simple_impl() {" << std::endl;
        simple_impl_desc_t d;
        if (!simple_impl_desc_init(prb_, &d)) return false;

        const int nfu = d.ndims_full_unroll;
        const int ldu = d.len_last_dim_unroll;
        const int n_jit_loops = prb_.ndims - d.ndims_full_unroll;
        assert(n_jit_loops <= ndims_jit_loop_max);

        xor_(reg_off_in, reg_off_in);
        xor_(reg_off_out, reg_off_out);
        if (prb_.scale_type == scale_type_t::MANY)
            xor_(reg_off_scale, reg_off_scale);

        Label l_loop[3];
        Reg64 reg_cnt[3] = {r15, r14, r13};

        if (n_jit_loops > 2)
            loop_begin(l_loop[2], reg_cnt[2], n(nfu + 2));

        if (n_jit_loops > 1)
            loop_begin(l_loop[1], reg_cnt[1], n(nfu + 1));

        if (n_jit_loops > 0)
            loop_begin(l_loop[0], reg_cnt[0], n(nfu + 0) / ldu);

        const bool optimized = false
            || process_direct_copy<avx>(d.len_unroll)
            || process_direct_copy<sse42>(d.len_unroll)
            || process_unroll_tr8x8(d.len_unroll);
        if (!optimized)
            process_unroll_generic(d.len_unroll);

        if (n_jit_loops > 0)
            loop_end(l_loop[0], reg_cnt[0],
                    n(nfu + 0) / ldu, is(nfu + 0) * ldu, os(nfu + 0) * ldu,
                    ss(nfu + 0) * ldu);

        if (n_jit_loops > 1)
            loop_end(l_loop[1], reg_cnt[1],
                    n(nfu + 1), is(nfu + 1), os(nfu + 1), ss(nfu + 1));

        if (n_jit_loops > 2)
            loop_end(l_loop[2], reg_cnt[2],
                    n(nfu + 2), is(nfu + 2), os(nfu + 2), ss(nfu + 2));

        return true;
    }

    void impl() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      void impl() {" << std::endl;
        if (simple_impl()) return;
        assert(!"no implementation available");
    }

    jit_uni_reorder_kernel_f32(const desc_t &desc)
        : kernel_t(desc), jit_generator(), bf16_emu_(nullptr) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          : kernel_t(desc), jit_generator(), bf16_emu_(nullptr) {" << std::endl;
        itype_sz = data_type_size(prb_.itype);
        otype_sz = data_type_size(prb_.otype);
        stype_sz = sizeof(float);
        is_cpx_ = mayiuse(avx512_core_bf16);
        if (prb_.otype == data_type::bf16 && !is_cpx_) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          if (prb_.otype == data_type::bf16 && !is_cpx_) {" << std::endl;
            bf16_emu_ = new bf16_emulation_t(this, vcvt_bf16_one, vcvt_bf16_eve,
                    vcvt_bf16_sel, reg_bf16_scratch, vcvt_bf16_tmp, vcvt_bf16_tmp);
            bf16_emu_->init_vcvtneps2bf16();
        }

        preamble();
#       define PARAM(x) ptr[abi_param1 + offsetof(call_param_t, x)]
        if (prb_.scale_type == scale_type_t::COMMON) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          if (prb_.scale_type == scale_type_t::COMMON) {" << std::endl;
            auto reg_ptr_scale_tmp = reg_ptr_in;
            mov(reg_ptr_scale_tmp, PARAM(scale));
            movups(xmm_scale, ptr[reg_ptr_scale_tmp]);
        } else if (prb_.scale_type == scale_type_t::MANY) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          } else if (prb_.scale_type == scale_type_t::MANY) {" << std::endl;
            mov(reg_ptr_scale, PARAM(scale));
        }
        mov(reg_ptr_in, PARAM(in));
        mov(reg_ptr_out, PARAM(out));
#       undef PARAM

        if (mayiuse(avx)) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          if (mayiuse(avx)) {" << std::endl;
            vxorps(xmm_zero, xmm_zero, xmm_zero);

            if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (prb_.itype == data_type::u8 && prb_.otype == data_type::s8) {" << std::endl;
                mov(reg_tmp.cvt32(), 0x7f7f7f7f);
                movd(xmm_4x127b, reg_tmp.cvt32());
            }
        }

        impl();
        postamble();
        ker_ = (void (*)(const call_param_t *))getCode();
    }
    ~jit_uni_reorder_kernel_f32() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      ~jit_uni_reorder_kernel_f32() {" << std::endl; delete bf16_emu_; }

private:
    int itype_sz;
    int otype_sz;
    int stype_sz;

    Reg64 reg_ptr_in = rsi;
    Reg64 reg_ptr_out = rdx;
    Reg64 reg_ptr_scale = abi_not_param1;

    Reg64 reg_off_in = r8;
    Reg64 reg_off_out = r9;
    Reg64 reg_off_scale = r10;

    Reg64 reg_tmp = rax;

    Xmm xmm_scale = xmm15;
    Xmm xmm_zero = xmm14;
    Xmm xmm_4x127b = xmm13; // TODO: unite with xmm_zero
    Xmm xmm_tmp = xmm12;

    /* bf16 support */
    bool is_cpx_;
    bf16_emulation_t *bf16_emu_;
    Reg64 reg_bf16_scratch = reg_tmp;
    Zmm vcvt_bf16_one = Zmm(16);
    Zmm vcvt_bf16_eve = Zmm(17);
    Zmm vcvt_bf16_sel = Zmm(18);
    Zmm vcvt_bf16_tmp = Zmm(19);
};

status_t kernel_t::desc_init(kernel_t::desc_t &desc, const prb_t &prb,
        int ndims_ker_max) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          int ndims_ker_max) {" << std::endl;
    desc.prb = prb;
    desc.prb.ioff = desc.prb.ooff = 0;

    if (ndims_ker_max > prb.ndims)
        return status::invalid_arguments;

    auto ndims_ker_max_f = [&]() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      auto ndims_ker_max_f = [&]() {" << std::endl;
        size_t cur_size = 1;
        for (int d = 0; d < prb.ndims; cur_size *= prb.nodes[d++].n)
            if (cur_size >= ker_prb_size_min) return d;
        return prb.ndims;
    };

    if (ndims_ker_max <= 0)
        ndims_ker_max = ndims_ker_max_f();

    /* traverse through kernel implementations */
    /* TODO: find a better way to do that... */
    desc.id = 0;
    for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      for (int ndims_ker = ndims_ker_max; ndims_ker > 0; --ndims_ker) {" << std::endl;
        desc.prb.ndims = ndims_ker;
        if (jit_uni_reorder_kernel_f32::applicable(desc.prb))
            return status::success;
    }

    return status::unimplemented;
}

kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:  kernel_t *kernel_t::create(const kernel_t::desc_t &desc) {" << std::endl;
    switch (desc.id) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      switch (desc.id) {" << std::endl;
    case 0: return new jit_uni_reorder_kernel_f32(desc);
    default: assert(!"unknown kernel id"); return nullptr;
    }

    return nullptr;
}

}

static void prb_block_for_cache(tr::prb_t &prb) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:  static void prb_block_for_cache(tr::prb_t &prb) {" << std::endl;
    if (prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > 16) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      if (prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > 16) {" << std::endl;
        /** an attempt to use caches more efficient and
         * address the 4K-aliasing issue */
        /* TODO: improve the logic around here */
        int j = 1;
        for (; j < prb.ndims && prb.nodes[j].is != 1; ++j);
        if (j == prb.ndims) return;

        /* it makes sense to re-prioritize sequential read over
         * sequential write if the former would not trash the
         * cache, i.e. is == 1 and os % 2^smth != 0. Smth is
         * set to 2 at the moment */
        const int move_to = prb.nodes[j].os % 4 != 0 ? 0 : 1;
        if (j == move_to) return;

        if (prb.nodes[j].n > 16 && prb.nodes[j].n % 16 == 0)
            prb_node_split(prb, j, 16);

        prb_node_move(prb, j, move_to);
        DEBUG({ printf("cache: "); prb_dump(prb); });
    }
}

/** finds the maximum number of dimension the kernel should process and
 * optionally splits one of the dimension to achieve better balance between
 * parallel driver and the kernel. */
static void prb_thread_kernel_balance(tr::prb_t &prb, int &ndims_ker_max) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:  static void prb_thread_kernel_balance(tr::prb_t &prb, int &ndims_ker_max) {" << std::endl;
    size_t sz_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        sz_total *= prb.nodes[d].n;

    /* sz_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t sz_drv_min = nstl::min<size_t>(
            16 * mkldnn_get_max_threads(),
            utils::div_up(sz_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * sz_ker_cur -- product of the dimension processed by a kernel
     * sz_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t sz_drv_cur = 1;
    for (; kdims > 1 && sz_drv_cur < sz_drv_min; --kdims)
        sz_drv_cur *= prb.nodes[kdims - 1].n;

    size_t sz_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        sz_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that sz_drv_cur >= sz_drv_min.
     *
     * It might happen that for chosen kdims the sz_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase sz_ker_cur. */
    bool want_borrow_ker_from_drv = true
        && kdims < prb.ndims
        && sz_ker_cur < tr::ker_prb_size_min
        && sz_drv_cur > sz_drv_min;
    if (want_borrow_ker_from_drv) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      if (want_borrow_ker_from_drv) {" << std::endl;
        /* sz_want_borrow is the minimal sz, so that:
         *  o) sz_ker_cur * sz_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     sz_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal sz_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t sz_want_borrow
            = utils::div_up(tr::ker_prb_size_min, sz_ker_cur);
        for (; prb.nodes[kdims].n % sz_want_borrow; ++sz_want_borrow);
        if (sz_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, sz_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the sz_drv_cur is too small (less than sz_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * sz_drv_cur. */
    bool want_borrow_drv_from_ker = true
        && sz_ker_cur > tr::ker_prb_size_min
        && sz_drv_cur < sz_drv_min;
    if (want_borrow_drv_from_ker) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      if (want_borrow_drv_from_ker) {" << std::endl;
        size_t sz_want_borrow = utils::div_up(sz_drv_min, sz_drv_cur);
        for (; prb.nodes[kdims - 1].n % sz_want_borrow; ++sz_want_borrow);
        if (sz_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(prb, kdims - 1,
                    prb.nodes[kdims - 1].n / sz_want_borrow);
    }

    ndims_ker_max = kdims;

    if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {" << std::endl;
        DEBUG({ printf("split: "); prb_dump(prb);
                printf("ndims_ker_max = %d\n", ndims_ker_max); });
    }
}

struct jit_uni_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        pd_t(const cpu_memory_pd_t *input_pd, const cpu_memory_pd_t *output_pd,
                const primitive_attr_t *attr)
            : cpu_reorder_pd_t(input_pd, output_pd, attr) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              : cpu_reorder_pd_t(input_pd, output_pd, attr) {" << std::endl;}

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                const memory_pd_t *input_pd, const memory_pd_t *output_pd,
                const primitive_attr_t *attr) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  const primitive_attr_t *attr) {" << std::endl;
            const memory_desc_t *imd = input_pd->desc();
            const memory_desc_t *omd = output_pd->desc();

            auto prb = tr::prb_t();

            if (imd->format == mkldnn_OhIw8o4i || imd->format == mkldnn_gOhIw8o4i ||
                imd->format == mkldnn_OhIw8o4i_s8s8 || imd->format == mkldnn_gOhIw8o4i_s8s8 ||
                omd->format == mkldnn_OhIw8o4i || omd->format == mkldnn_gOhIw8o4i ||
                omd->format == mkldnn_OhIw8o4i_s8s8 || omd->format == mkldnn_gOhIw8o4i_s8s8 ||
                omd->format == mkldnn_OdhIw8o4i || omd->format == mkldnn_gOdhIw8o4i ||
                omd->format == mkldnn_OdhIw8o4i_s8s8 || omd->format == mkldnn_gOdhIw8o4i_s8s8 ||
                omd->format == mkldnn_dhwio || omd->format == mkldnn_dhwigo ||
                omd->format == mkldnn_hwio || omd->format == mkldnn_hwigo ||
                omd->format == mkldnn_dhwio_s8s8 || omd->format == mkldnn_dhwigo_s8s8)
                return status::unimplemented;

            status_t prb_init_status = prb_init(prb, *imd, *omd, attr);
            if (prb_init_status != success) return prb_init_status;

            DEBUG({ printf("init : "); prb_dump(prb); });
            prb_normalize(prb);
            DEBUG({ printf("norm : "); prb_dump(prb); });
            prb_simplify(prb);
            DEBUG({ printf("smpl : "); prb_dump(prb); });

            prb_block_for_cache(prb);

            int ndims_ker_max;
            prb_thread_kernel_balance(prb, ndims_ker_max);

            tr::kernel_t::desc_t ker_desc;
            status_t ker_init_status
                = tr::kernel_t::desc_init(ker_desc, prb, ndims_ker_max);
            if (ker_init_status != status::success) return ker_init_status;

            const int ndims_driver = prb.ndims - ker_desc.prb.ndims;
            if (ndims_driver > jit_uni_reorder_t::ndims_driver_max)
                return status::unimplemented;

            DEBUG({ printf("ker  : "); prb_dump(ker_desc.prb); });

            auto _pd = new pd_t((const cpu_memory_pd_t *)input_pd,
                    (const cpu_memory_pd_t *)output_pd, attr);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              if (_pd->init() != success) {" << std::endl; delete _pd; return unimplemented; }
            _pd->prb_ = prb;
            _pd->ker_desc_ = ker_desc;
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        tr::prb_t prb_;
        tr::kernel_t::desc_t ker_desc_;
    };

    jit_uni_reorder_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          : cpu_primitive_t(apd, inputs, outputs) {" << std::endl;
        kernel_ = tr::kernel_t::create(pd()->ker_desc_);
        assert(kernel_);
    }
    ~jit_uni_reorder_t() {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:      ~jit_uni_reorder_t() {" << std::endl; delete kernel_; }

    void omp_driver_0d(int off, const char *in, char *out,
            const float *scale) const {
        tr::call_param_t c{in, out, scale};
        (*kernel_)(&c);
    }

    void omp_driver_1d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          for_nd(ithr, nthr, (ptrdiff_t)ns[0].n, [&](ptrdiff_t d0) {" << std::endl;
            auto c = tr::call_param_t();
            c.in = in + d0 * ns[0].is * data_type_size(pd()->prb_.itype);
            c.out = out + d0 * ns[0].os * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss;
            (*kernel_)(&c);
        });
    }

    void omp_driver_2d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d1, ptrdiff_t d0) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  [&](ptrdiff_t d1, ptrdiff_t d0) {" << std::endl;
            auto c = tr::call_param_t();
            c.in = in + (d0 * ns[0].is + d1 * ns[1].is)
                * data_type_size(pd()->prb_.itype);
            c.out = out + (d0 * ns[0].os + d1 * ns[1].os)
                * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss;
            (*kernel_)(&c);
        });
    }

    void omp_driver_3d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[2].n, (ptrdiff_t)ns[1].n,
                (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  [&](ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {" << std::endl;
            auto c = tr::call_param_t();
            c.in = in + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is)
                * data_type_size(pd()->prb_.itype);
            c.out = out + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os)
                * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss;
            (*kernel_)(&c);
        });
    }

    void omp_driver_4d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const {
        const tr::node_t *ns = pd()->prb_.nodes + off;
        for_nd(ithr, nthr, (ptrdiff_t)ns[3].n, (ptrdiff_t)ns[2].n,
                (ptrdiff_t)ns[1].n, (ptrdiff_t)ns[0].n,
                [&](ptrdiff_t d3, ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  [&](ptrdiff_t d3, ptrdiff_t d2, ptrdiff_t d1, ptrdiff_t d0) {" << std::endl;
            auto c = tr::call_param_t();
            c.in = in + (d0 * ns[0].is + d1 * ns[1].is + d2 * ns[2].is
                    + d3 * ns[3].is) * data_type_size(pd()->prb_.itype);
            c.out = out + (d0 * ns[0].os + d1 * ns[1].os + d2 * ns[2].os
                    + d3 * ns[3].os) * data_type_size(pd()->prb_.otype);
            c.scale = scale + d0 * ns[0].ss + d1 * ns[1].ss + d2 * ns[2].ss
                + d3 * ns[3].ss;
            (*kernel_)(&c);
        });
    }

    void omp_driver(const char *in, char *out, const float *scale) const {
        in += pd()->prb_.ioff * data_type_size(pd()->prb_.itype);
        out += pd()->prb_.ooff * data_type_size(pd()->prb_.otype);

        DEBUG({ printf("prb : "); tr::prb_dump(pd()->prb_); });
        DEBUG({ printf("ker : "); tr::prb_dump(pd()->ker_desc_.prb); });

        int ndims = pd()->prb_.ndims;
        int ndims_ker = pd()->ker_desc_.prb.ndims;
        assert(ndims - ndims_ker <= ndims_driver_max);

        if (ndims - ndims_ker == 0) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          if (ndims - ndims_ker == 0) {" << std::endl;
            set_rnd_mode(pd()->attr()->round_mode_);
            omp_driver_0d(ndims_ker, in, out, scale);
            restore_rnd_mode();
        } else {
            size_t work_amount = 0;
            const tr::node_t *ns = pd()->prb_.nodes + ndims_ker;
            switch (ndims - ndims_ker) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              switch (ndims - ndims_ker) {" << std::endl;
                case 1: work_amount = (size_t)ns[0].n; break;
                case 2: work_amount = (size_t)ns[1].n * (size_t)ns[0].n; break;
                case 3: work_amount = (size_t)ns[2].n * (size_t)ns[1].n * (size_t)ns[0].n; break;
                case 4: work_amount = (size_t)ns[3].n * (size_t)ns[2].n * (size_t)ns[1].n * (size_t)ns[0].n; break;
                default: assert(!"unimplemented");
            }

            parallel(0, work_amount, [&](const int ithr, const int nthr) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:              parallel(0, work_amount, [&](const int ithr, const int nthr) {" << std::endl;
                set_rnd_mode(pd()->attr()->round_mode_);
                switch (ndims - ndims_ker) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:                  switch (ndims - ndims_ker) {" << std::endl;
                case 1: omp_driver_1d(ithr, nthr, ndims_ker, in, out, scale); break;
                case 2: omp_driver_2d(ithr, nthr, ndims_ker, in, out, scale); break;
                case 3: omp_driver_3d(ithr, nthr, ndims_ker, in, out, scale); break;
                case 4: omp_driver_4d(ithr, nthr, ndims_ker, in, out, scale); break;
                default: assert(!"unimplemented");
                }
                restore_rnd_mode();
            });
        }
    }

    virtual void execute(event_t *e) const {
        auto in = reinterpret_cast<const char *>(input_memory(0));
        auto out = reinterpret_cast<char *>(memory());

        omp_driver(in, out, pd()->attr()->output_scales_.scales_);

        e->set_state(event_t::ready);
    }

    enum { ndims_driver_max = 4 };

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    tr::kernel_t *kernel_;
};

status_t jit_uni_reorder_create(reorder_pd_t **reorder_pd,
        const memory_pd_t *input_pd, const memory_pd_t *output_pd,
        const primitive_attr_t *attr) {
    std::cerr << "./inference-engine/thirdparty/mkl-dnn/src/cpu/jit_uni_reorder.cpp:          const primitive_attr_t *attr) {" << std::endl;
    return jit_uni_reorder_t::pd_t::create(reorder_pd, input_pd, output_pd,
            attr);
}

}
}
}
