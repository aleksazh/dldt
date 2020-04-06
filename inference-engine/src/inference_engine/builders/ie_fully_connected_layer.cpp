#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_fully_connected_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::FullyConnectedLayer::FullyConnectedLayer(const std::string& name): LayerDecorator("FullyConnected", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp:  Builder::FullyConnectedLayer::FullyConnectedLayer(const std::string& name): LayerDecorator('FullyConnected', name) {" << std::endl;
    getLayer()->getInputPorts().resize(3);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getInputPorts()[2].setParameter("type", "biases");
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getParameters()["out-size"] = 0;
}

Builder::FullyConnectedLayer::FullyConnectedLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp:  Builder::FullyConnectedLayer::FullyConnectedLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("FullyConnected");
}

Builder::FullyConnectedLayer::FullyConnectedLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp:  Builder::FullyConnectedLayer::FullyConnectedLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("FullyConnected");
}

Builder::FullyConnectedLayer &Builder::FullyConnectedLayer::setName(const std::string &name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp:  Builder::FullyConnectedLayer &Builder::FullyConnectedLayer::setName(const std::string &name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::FullyConnectedLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setInputPort(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp:  Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setInputPort(const Port& port) {" << std::endl;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

const Port& Builder::FullyConnectedLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setOutputPort(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp:  Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setOutputPort(const Port& port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::FullyConnectedLayer::getOutputNum() const {
    return getLayer()->getParameters().at("out-size");
}

Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setOutputNum(size_t outNum) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp:  Builder::FullyConnectedLayer& Builder::FullyConnectedLayer::setOutputNum(size_t outNum) {" << std::endl;
    getLayer()->getParameters()["out-size"] = outNum;
    return *this;
}

REG_VALIDATOR_FOR(FullyConnected, [](const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp:  REG_VALIDATOR_FOR(FullyConnected, [](const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {" << std::endl;
});

REG_CONVERTER_FOR(FullyConnected, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_fully_connected_layer.cpp:  REG_CONVERTER_FOR(FullyConnected, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["out-size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("out-size", 0));
});
