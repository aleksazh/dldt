#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_layers_dispatcher.hpp"
#include <details/ie_exception.hpp>
#include "nodes/list.hpp"
#include <cpu_isa_traits.hpp>
#include <memory>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

void addDefaultExtensions(MKLDNNExtensionManager::Ptr mngr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_layers_dispatcher.cpp:  void addDefaultExtensions(MKLDNNExtensionManager::Ptr mngr) {" << std::endl;
    if (!mngr)
        THROW_IE_EXCEPTION << "Cannot add default extensions! Extension manager is empty.";

    if (mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::cpu_isa_t::avx512_common)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_layers_dispatcher.cpp:      if (mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::cpu_isa_t::avx512_common)) {" << std::endl;
        auto platformExtensions = std::make_shared<Extensions::Cpu::MKLDNNExtensions<mkldnn::impl::cpu::cpu_isa_t::avx512_common>>();
        mngr->AddExtension(platformExtensions);
    }
    if (mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::cpu_isa_t::avx2)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_layers_dispatcher.cpp:      if (mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::cpu_isa_t::avx2)) {" << std::endl;
        auto platformExtensions = std::make_shared<Extensions::Cpu::MKLDNNExtensions<mkldnn::impl::cpu::cpu_isa_t::avx2>>();
        mngr->AddExtension(platformExtensions);
    }
    if (mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::cpu_isa_t::sse42)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_layers_dispatcher.cpp:      if (mkldnn::impl::cpu::mayiuse(mkldnn::impl::cpu::cpu_isa_t::sse42)) {" << std::endl;
        auto platformExtensions = std::make_shared<Extensions::Cpu::MKLDNNExtensions<mkldnn::impl::cpu::cpu_isa_t::sse42>>();
        mngr->AddExtension(platformExtensions);
    }

    auto defaultExtensions = std::make_shared<Extensions::Cpu::MKLDNNExtensions<mkldnn::impl::cpu::cpu_isa_t::isa_any>>();
    mngr->AddExtension(defaultExtensions);
}

}  // namespace MKLDNNPlugin
