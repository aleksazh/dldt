#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_reorg_yolo_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ReorgYoloLayer::ReorgYoloLayer(const std::string& name): LayerDecorator("ReorgYolo", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp:  Builder::ReorgYoloLayer::ReorgYoloLayer(const std::string& name): LayerDecorator('ReorgYolo', name) {" << std::endl;
    getLayer()->getInputPorts().resize(1);
    getLayer()->getOutputPorts().resize(1);
}

Builder::ReorgYoloLayer::ReorgYoloLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp:  Builder::ReorgYoloLayer::ReorgYoloLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ReorgYolo");
}

Builder::ReorgYoloLayer::ReorgYoloLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp:  Builder::ReorgYoloLayer::ReorgYoloLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ReorgYolo");
}

Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp:  Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}
const Port& Builder::ReorgYoloLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}
Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setInputPort(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp:  Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setInputPort(const Port& port) {" << std::endl;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
const Port& Builder::ReorgYoloLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}
Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setOutputPort(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp:  Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setOutputPort(const Port& port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}
int Builder::ReorgYoloLayer::getStride() const {
    return getLayer()->getParameters().at("stride");
}
Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setStride(int stride) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp:  Builder::ReorgYoloLayer& Builder::ReorgYoloLayer::setStride(int stride) {" << std::endl;
    getLayer()->getParameters()["stride"] = stride;
    return *this;
}

REG_CONVERTER_FOR(ReorgYolo, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_reorg_yolo_layer.cpp:  REG_CONVERTER_FOR(ReorgYolo, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["stride"] = cnnLayer->GetParamAsInt("stride", 0);
});