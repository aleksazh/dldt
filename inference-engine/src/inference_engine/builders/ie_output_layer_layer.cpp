#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_output_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::OutputLayer::OutputLayer(const std::string& name): LayerDecorator("Output", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_output_layer_layer.cpp:  Builder::OutputLayer::OutputLayer(const std::string& name): LayerDecorator('Output', name) {" << std::endl;
    getLayer()->getInputPorts().resize(1);
}

Builder::OutputLayer::OutputLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_output_layer_layer.cpp:  Builder::OutputLayer::OutputLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Output");
}

Builder::OutputLayer::OutputLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_output_layer_layer.cpp:  Builder::OutputLayer::OutputLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Output");
}

Builder::OutputLayer& Builder::OutputLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_output_layer_layer.cpp:  Builder::OutputLayer& Builder::OutputLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::OutputLayer::getPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::OutputLayer& Builder::OutputLayer::setPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_output_layer_layer.cpp:  Builder::OutputLayer& Builder::OutputLayer::setPort(const Port &port) {" << std::endl;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

REG_VALIDATOR_FOR(Output, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_output_layer_layer.cpp:  REG_VALIDATOR_FOR(Output, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {" << std::endl;});
