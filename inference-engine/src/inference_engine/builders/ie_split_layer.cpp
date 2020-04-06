#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_cnn_layer_builder.h>

#include <builders/ie_split_layer.hpp>
#include <string>
#include <vector>

using namespace InferenceEngine;

Builder::SplitLayer::SplitLayer(const std::string& name): LayerDecorator("Split", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_split_layer.cpp:  Builder::SplitLayer::SplitLayer(const std::string& name): LayerDecorator('Split', name) {" << std::endl;
    getLayer()->getInputPorts().resize(1);
    setAxis(1);
}

Builder::SplitLayer::SplitLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_split_layer.cpp:  Builder::SplitLayer::SplitLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Split");
}

Builder::SplitLayer::SplitLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_split_layer.cpp:  Builder::SplitLayer::SplitLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Split");
}

Builder::SplitLayer& Builder::SplitLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_split_layer.cpp:  Builder::SplitLayer& Builder::SplitLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::SplitLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::SplitLayer& Builder::SplitLayer::setInputPort(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_split_layer.cpp:  Builder::SplitLayer& Builder::SplitLayer::setInputPort(const Port& port) {" << std::endl;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

const std::vector<Port>& Builder::SplitLayer::getOutputPorts() const {
    return getLayer()->getOutputPorts();
}

Builder::SplitLayer& Builder::SplitLayer::setOutputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_split_layer.cpp:  Builder::SplitLayer& Builder::SplitLayer::setOutputPorts(const std::vector<Port>& ports) {" << std::endl;
    getLayer()->getOutputPorts() = ports;
    return *this;
}

size_t Builder::SplitLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}

Builder::SplitLayer& Builder::SplitLayer::setAxis(size_t axis) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_split_layer.cpp:  Builder::SplitLayer& Builder::SplitLayer::setAxis(size_t axis) {" << std::endl;
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}

REG_CONVERTER_FOR(Split, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_split_layer.cpp:  REG_CONVERTER_FOR(Split, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis", 1));
});