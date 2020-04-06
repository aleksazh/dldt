#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_relu6_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::ReLU6Layer::ReLU6Layer(const std::string& name): LayerDecorator("ReLU6", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp:  Builder::ReLU6Layer::ReLU6Layer(const std::string& name): LayerDecorator('ReLU6', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setN(6);
}

Builder::ReLU6Layer::ReLU6Layer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp:  Builder::ReLU6Layer::ReLU6Layer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ReLU6");
}

Builder::ReLU6Layer::ReLU6Layer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp:  Builder::ReLU6Layer::ReLU6Layer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ReLU6");
}

Builder::ReLU6Layer& Builder::ReLU6Layer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp:  Builder::ReLU6Layer& Builder::ReLU6Layer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ReLU6Layer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ReLU6Layer& Builder::ReLU6Layer::setPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp:  Builder::ReLU6Layer& Builder::ReLU6Layer::setPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

float Builder::ReLU6Layer::getN() const {
    return getLayer()->getParameters().at("n");
}

Builder::ReLU6Layer& Builder::ReLU6Layer::setN(float n) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp:  Builder::ReLU6Layer& Builder::ReLU6Layer::setN(float n) {" << std::endl;
    getLayer()->getParameters()["n"] = n;
    return *this;
}

REG_VALIDATOR_FOR(ReLU6, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp:  REG_VALIDATOR_FOR(ReLU6, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {" << std::endl;
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp:          input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {" << std::endl;
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});

REG_CONVERTER_FOR(ReLU6, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_relu6_layer.cpp:  REG_CONVERTER_FOR(ReLU6, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["n"] = cnnLayer->GetParamAsFloat("n", 0);
});
