#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_input_layer.hpp>
#include <string>

using namespace InferenceEngine;

Builder::InputLayer::InputLayer(const std::string& name): LayerDecorator("Input", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_input_layer_layer.cpp:  Builder::InputLayer::InputLayer(const std::string& name): LayerDecorator('Input', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
}

Builder::InputLayer::InputLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_input_layer_layer.cpp:  Builder::InputLayer::InputLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Input");
}

Builder::InputLayer::InputLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_input_layer_layer.cpp:  Builder::InputLayer::InputLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Input");
}

Builder::InputLayer& Builder::InputLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_input_layer_layer.cpp:  Builder::InputLayer& Builder::InputLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::InputLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::InputLayer& Builder::InputLayer::setPort(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_input_layer_layer.cpp:  Builder::InputLayer& Builder::InputLayer::setPort(const Port& port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

REG_VALIDATOR_FOR(Input, [](const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_input_layer_layer.cpp:  REG_VALIDATOR_FOR(Input, [](const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {" << std::endl;
    if (layer->getOutputPorts()[0].shape().empty())
        THROW_IE_EXCEPTION << layer->getType() << " node " << layer->getName() << " should have shape!";
});
