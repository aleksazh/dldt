#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_softmax_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::SoftMaxLayer::SoftMaxLayer(const std::string& name): LayerDecorator("SoftMax", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_softmax_layer.cpp:  Builder::SoftMaxLayer::SoftMaxLayer(const std::string& name): LayerDecorator('SoftMax', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setAxis(1);
}

Builder::SoftMaxLayer::SoftMaxLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_softmax_layer.cpp:  Builder::SoftMaxLayer::SoftMaxLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("SoftMax");
}

Builder::SoftMaxLayer::SoftMaxLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_softmax_layer.cpp:  Builder::SoftMaxLayer::SoftMaxLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("SoftMax");
}

Builder::SoftMaxLayer& Builder::SoftMaxLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_softmax_layer.cpp:  Builder::SoftMaxLayer& Builder::SoftMaxLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::SoftMaxLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::SoftMaxLayer& Builder::SoftMaxLayer::setPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_softmax_layer.cpp:  Builder::SoftMaxLayer& Builder::SoftMaxLayer::setPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

size_t Builder::SoftMaxLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}

Builder::SoftMaxLayer& Builder::SoftMaxLayer::setAxis(size_t axis) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_softmax_layer.cpp:  Builder::SoftMaxLayer& Builder::SoftMaxLayer::setAxis(size_t axis) {" << std::endl;
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}

REG_CONVERTER_FOR(SoftMax, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_softmax_layer.cpp:  REG_CONVERTER_FOR(SoftMax, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis", 1));
});