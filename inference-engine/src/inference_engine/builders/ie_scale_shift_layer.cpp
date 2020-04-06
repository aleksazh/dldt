#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_scale_shift_layer.hpp>

#include <string>

using namespace InferenceEngine;

Builder::ScaleShiftLayer::ScaleShiftLayer(const std::string& name): LayerDecorator("ScaleShift", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_scale_shift_layer.cpp:  Builder::ScaleShiftLayer::ScaleShiftLayer(const std::string& name): LayerDecorator('ScaleShift', name) {" << std::endl;
    getLayer()->getInputPorts().resize(3);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getInputPorts()[2].setParameter("type", "biases");
    getLayer()->getOutputPorts().resize(1);
}

Builder::ScaleShiftLayer::ScaleShiftLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_scale_shift_layer.cpp:  Builder::ScaleShiftLayer::ScaleShiftLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ScaleShift");
}

Builder::ScaleShiftLayer::ScaleShiftLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_scale_shift_layer.cpp:  Builder::ScaleShiftLayer::ScaleShiftLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ScaleShift");
}

Builder::ScaleShiftLayer& Builder::ScaleShiftLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_scale_shift_layer.cpp:  Builder::ScaleShiftLayer& Builder::ScaleShiftLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ScaleShiftLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ScaleShiftLayer& Builder::ScaleShiftLayer::setPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_scale_shift_layer.cpp:  Builder::ScaleShiftLayer& Builder::ScaleShiftLayer::setPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}