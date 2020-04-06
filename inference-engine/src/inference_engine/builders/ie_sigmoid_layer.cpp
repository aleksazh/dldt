#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_sigmoid_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::SigmoidLayer::SigmoidLayer(const std::string& name): LayerDecorator("Sigmoid", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_sigmoid_layer.cpp:  Builder::SigmoidLayer::SigmoidLayer(const std::string& name): LayerDecorator('Sigmoid', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::SigmoidLayer::SigmoidLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_sigmoid_layer.cpp:  Builder::SigmoidLayer::SigmoidLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Sigmoid");
}

Builder::SigmoidLayer::SigmoidLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_sigmoid_layer.cpp:  Builder::SigmoidLayer::SigmoidLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Sigmoid");
}

Builder::SigmoidLayer& Builder::SigmoidLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_sigmoid_layer.cpp:  Builder::SigmoidLayer& Builder::SigmoidLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::SigmoidLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::SigmoidLayer& Builder::SigmoidLayer::setPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_sigmoid_layer.cpp:  Builder::SigmoidLayer& Builder::SigmoidLayer::setPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
