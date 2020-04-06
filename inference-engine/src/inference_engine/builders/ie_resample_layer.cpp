#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_resample_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ResampleLayer::ResampleLayer(const std::string& name): LayerDecorator("Resample", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer::ResampleLayer(const std::string& name): LayerDecorator('Resample', name) {" << std::endl;
    getLayer()->getInputPorts().resize(1);
    getLayer()->getOutputPorts().resize(1);
}

Builder::ResampleLayer::ResampleLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer::ResampleLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Resample");
}

Builder::ResampleLayer::ResampleLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer::ResampleLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Resample");
}

Builder::ResampleLayer& Builder::ResampleLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer& Builder::ResampleLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}
const Port& Builder::ResampleLayer::getInputPort() const {
    return getLayer()->getInputPorts()[0];
}
Builder::ResampleLayer& Builder::ResampleLayer::setInputPort(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer& Builder::ResampleLayer::setInputPort(const Port& port) {" << std::endl;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}
const Port& Builder::ResampleLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}
Builder::ResampleLayer& Builder::ResampleLayer::setOutputPort(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer& Builder::ResampleLayer::setOutputPort(const Port& port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::string &Builder::ResampleLayer::getResampleType() const {
    return getLayer()->getParameters().at("type");
}

Builder::ResampleLayer &Builder::ResampleLayer::setResampleType(const std::string &type) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer &Builder::ResampleLayer::setResampleType(const std::string &type) {" << std::endl;
    getLayer()->getParameters()["type"] = type;
    return *this;
}

bool Builder::ResampleLayer::getAntialias() const {
    return getLayer()->getParameters().at("antialias");
}

Builder::ResampleLayer &Builder::ResampleLayer::setAntialias(bool antialias) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer &Builder::ResampleLayer::setAntialias(bool antialias) {" << std::endl;
    getLayer()->getParameters()["antialias"] = antialias;
    return *this;
}

float Builder::ResampleLayer::getFactor() const {
    return getLayer()->getParameters().at("factor");
}

Builder::ResampleLayer &Builder::ResampleLayer::setFactor(float factor) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer &Builder::ResampleLayer::setFactor(float factor) {" << std::endl;
    getLayer()->getParameters()["factor"] = factor;
    return *this;
}

size_t Builder::ResampleLayer::getWidth() const {
    return getLayer()->getParameters().at("width");
}

Builder::ResampleLayer &Builder::ResampleLayer::setWidth(size_t width) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer &Builder::ResampleLayer::setWidth(size_t width) {" << std::endl;
    getLayer()->getParameters()["width"] = width;
    return *this;
}

size_t Builder::ResampleLayer::getHeight() const {
    return getLayer()->getParameters().at("height");
}

Builder::ResampleLayer &Builder::ResampleLayer::setHeight(size_t height) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  Builder::ResampleLayer &Builder::ResampleLayer::setHeight(size_t height) {" << std::endl;
    getLayer()->getParameters()["height"] = height;
    return *this;
}

REG_CONVERTER_FOR(Resample, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_resample_layer.cpp:  REG_CONVERTER_FOR(Resample, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["height"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("height", 0));
    layer.getParameters()["width"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("width", 0));
    layer.getParameters()["factor"] = cnnLayer->GetParamAsFloat("factor", 0);
    layer.getParameters()["antialias"] = cnnLayer->GetParamAsBool("antialias", false);
    layer.getParameters()["type"] = cnnLayer->GetParamAsString("type");
});