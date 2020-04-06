#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_memory_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::MemoryLayer::MemoryLayer(const std::string& name): LayerDecorator("Memory", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  Builder::MemoryLayer::MemoryLayer(const std::string& name): LayerDecorator('Memory', name) {" << std::endl;
    setSize(2);
}

Builder::MemoryLayer::MemoryLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  Builder::MemoryLayer::MemoryLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Memory");
}

Builder::MemoryLayer::MemoryLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  Builder::MemoryLayer::MemoryLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Memory");
}

Builder::MemoryLayer& Builder::MemoryLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  Builder::MemoryLayer& Builder::MemoryLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::MemoryLayer::getInputPort() const {
    if (getLayer()->getInputPorts().empty()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:      if (getLayer()->getInputPorts().empty()) {" << std::endl;
        THROW_IE_EXCEPTION << "No inputs ports for layer: " << getLayer()->getName();
    }
    return getLayer()->getInputPorts()[0];
}

Builder::MemoryLayer& Builder::MemoryLayer::setInputPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  Builder::MemoryLayer& Builder::MemoryLayer::setInputPort(const Port &port) {" << std::endl;
    getLayer()->getInputPorts().resize(1);
    getLayer()->getInputPorts()[0] = port;
    setIndex(0);
    return *this;
}

const Port& Builder::MemoryLayer::getOutputPort() const {
    if (getLayer()->getOutputPorts().empty()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:      if (getLayer()->getOutputPorts().empty()) {" << std::endl;
        THROW_IE_EXCEPTION << "No output ports for layer: " << getLayer()->getName();
    }
    return getLayer()->getOutputPorts()[0];
}

Builder::MemoryLayer& Builder::MemoryLayer::setOutputPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  Builder::MemoryLayer& Builder::MemoryLayer::setOutputPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getOutputPorts()[0] = port;
    setIndex(1);
    return *this;
}

const std::string Builder::MemoryLayer::getId() const {
    return getLayer()->getParameters().at("id");
}
Builder::MemoryLayer& Builder::MemoryLayer::setId(const std::string& id) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  Builder::MemoryLayer& Builder::MemoryLayer::setId(const std::string& id) {" << std::endl;
    getLayer()->getParameters()["id"] = id;
    return *this;
}
size_t Builder::MemoryLayer::getIndex() const {
    return getLayer()->getParameters().at("index");
}
Builder::MemoryLayer& Builder::MemoryLayer::setIndex(size_t index) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  Builder::MemoryLayer& Builder::MemoryLayer::setIndex(size_t index) {" << std::endl;
    if (index > 1)
        THROW_IE_EXCEPTION << "Index supports only 0 and 1 values.";
    getLayer()->getParameters()["index"] = index;
    return *this;
}
size_t Builder::MemoryLayer::getSize() const {
    return getLayer()->getParameters().at("size");
}
Builder::MemoryLayer& Builder::MemoryLayer::setSize(size_t size) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  Builder::MemoryLayer& Builder::MemoryLayer::setSize(size_t size) {" << std::endl;
    if (size != 2)
        THROW_IE_EXCEPTION << "Only size equal 2 is supported.";
    getLayer()->getParameters()["size"] = size;
    return *this;
}
REG_VALIDATOR_FOR(Memory, [](const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  REG_VALIDATOR_FOR(Memory, [](const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {" << std::endl;
});

REG_CONVERTER_FOR(Memory, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_memory_layer.cpp:  REG_CONVERTER_FOR(Memory, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["id"] = cnnLayer->GetParamAsString("id", 0);
    layer.getParameters()["index"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("index", 0));
    layer.getParameters()["size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("size", 0));
});

