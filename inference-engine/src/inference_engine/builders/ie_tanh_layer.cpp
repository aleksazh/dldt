#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_tanh_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::TanHLayer::TanHLayer(const std::string& name): LayerDecorator("TanH", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_tanh_layer.cpp:  Builder::TanHLayer::TanHLayer(const std::string& name): LayerDecorator('TanH', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::TanHLayer::TanHLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_tanh_layer.cpp:  Builder::TanHLayer::TanHLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("TanH");
}

Builder::TanHLayer::TanHLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_tanh_layer.cpp:  Builder::TanHLayer::TanHLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("TanH");
}

Builder::TanHLayer& Builder::TanHLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_tanh_layer.cpp:  Builder::TanHLayer& Builder::TanHLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::TanHLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::TanHLayer& Builder::TanHLayer::setPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_tanh_layer.cpp:  Builder::TanHLayer& Builder::TanHLayer::setPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

REG_VALIDATOR_FOR(TanH, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_tanh_layer.cpp:  REG_VALIDATOR_FOR(TanH, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {" << std::endl;
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_tanh_layer.cpp:          input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {" << std::endl;
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});
