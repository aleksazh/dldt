#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_plugin_internal.hpp"
#ifdef ENABLE_NGRAPH
#include "cnn_network_ngraph_impl.hpp"
#endif
#include <memory>

std::shared_ptr<InferenceEngine::ICNNNetwork> InferenceEngine::InferencePluginInternal::ConvertAndCloneNetwork(ICNNNetwork& network) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/impl/ie_plugin_internal.cpp:  std::shared_ptr<InferenceEngine::ICNNNetwork> InferenceEngine::InferencePluginInternal::ConvertAndCloneNetwork(ICNNNetwork& network) {" << std::endl;
#ifdef ENABLE_NGRAPH
    if (auto networkNGraph = dynamic_cast<CNNNetworkNGraphImpl*>(&network)) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/impl/ie_plugin_internal.cpp:      if (auto networkNGraph = dynamic_cast<CNNNetworkNGraphImpl*>(&network)) {" << std::endl;
        auto nGraphNetwork = networkNGraph->cloneNGraphImpl();
        nGraphNet = nGraphNetwork;
        return nGraphNetwork->getCNNNetwork();
    }
#endif
    return CloneNetwork(network);
}
