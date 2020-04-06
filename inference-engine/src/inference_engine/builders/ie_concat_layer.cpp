#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_concat_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ConcatLayer::ConcatLayer(const std::string& name): LayerDecorator("Concat", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:  Builder::ConcatLayer::ConcatLayer(const std::string& name): LayerDecorator('Concat', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    setAxis(1);
}

Builder::ConcatLayer::ConcatLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:  Builder::ConcatLayer::ConcatLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Concat");
}

Builder::ConcatLayer::ConcatLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:  Builder::ConcatLayer::ConcatLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Concat");
}

Builder::ConcatLayer& Builder::ConcatLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:  Builder::ConcatLayer& Builder::ConcatLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ConcatLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ConcatLayer& Builder::ConcatLayer::setOutputPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:  Builder::ConcatLayer& Builder::ConcatLayer::setOutputPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

const std::vector<Port>& Builder::ConcatLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::ConcatLayer& Builder::ConcatLayer::setInputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:  Builder::ConcatLayer& Builder::ConcatLayer::setInputPorts(const std::vector<Port>& ports) {" << std::endl;
    getLayer()->getInputPorts() = ports;
    return *this;
}

size_t Builder::ConcatLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}

Builder::ConcatLayer& Builder::ConcatLayer::setAxis(size_t axis) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:  Builder::ConcatLayer& Builder::ConcatLayer::setAxis(size_t axis) {" << std::endl;
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}

REG_VALIDATOR_FOR(Concat, [] (const InferenceEngine::Builder::Layer::CPtr &input_layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:  REG_VALIDATOR_FOR(Concat, [] (const InferenceEngine::Builder::Layer::CPtr &input_layer, bool partial) {" << std::endl;
    if (partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:      if (partial) {" << std::endl;
        return;
    }
    Builder::ConcatLayer layer(input_layer);
    if (layer.getInputPorts().size() < 1) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:      if (layer.getInputPorts().size() < 1) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input ports. "
                           << "It takes at least two Blobs";
    }
    for (size_t i = 1; i < layer.getInputPorts().size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:      for (size_t i = 1; i < layer.getInputPorts().size(); ++i) {" << std::endl;
        if (layer.getInputPorts()[i - 1].shape().size() != layer.getInputPorts()[i].shape().size()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:          if (layer.getInputPorts()[i - 1].shape().size() != layer.getInputPorts()[i].shape().size()) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input ports. "
                               << "It should have equal number of dimensions";
        }
    }
    if (layer.getInputPorts()[0].shape().size() != layer.getOutputPort().shape().size()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:      if (layer.getInputPorts()[0].shape().size() != layer.getOutputPort().shape().size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input and output ports "
                           << "It should have equal number of dimensions";
    }
    if (layer.getAxis() >= layer.getOutputPort().shape().size()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:      if (layer.getAxis() >= layer.getOutputPort().shape().size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer " << layer.getName() << "contains incorrect axis. "
                           << "It should be >= 0 and < number of port's dimensions.";
    }
    for (size_t i = 0; i < layer.getOutputPort().shape().size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:      for (size_t i = 0; i < layer.getOutputPort().shape().size(); ++i) {" << std::endl;
        if (i == layer.getAxis()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:          if (i == layer.getAxis()) {" << std::endl;
            size_t sumInputDimensions = 0;
            for (const Port& port : layer.getInputPorts()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:              for (const Port& port : layer.getInputPorts()) {" << std::endl;
                sumInputDimensions += port.shape()[i];
            }
            if (sumInputDimensions != layer.getOutputPort().shape()[i]) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:              if (sumInputDimensions != layer.getOutputPort().shape()[i]) {" << std::endl;
                THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input and output ports "
                                   << "Sum of input port's dimensions in the given axis should be equal to output ports dimension in the same axis.";
            }
        } else {
            for (const Port& port : layer.getInputPorts()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:              for (const Port& port : layer.getInputPorts()) {" << std::endl;
                if (port.shape()[i] != layer.getOutputPort().shape()[i]) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:                  if (port.shape()[i] != layer.getOutputPort().shape()[i]) {" << std::endl;
                    THROW_IE_EXCEPTION << "Layer " << layer.getName() << " contains incorrect input and output ports. "
                                       << "It should have equal dimensions in axis different from given";
                }
            }
        }
    }
});

REG_CONVERTER_FOR(Concat, [] (const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp:  REG_CONVERTER_FOR(Concat, [] (const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["axis"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("axis", 1));
});

