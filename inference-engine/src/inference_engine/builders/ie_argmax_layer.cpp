#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_argmax_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ArgMaxLayer::ArgMaxLayer(const std::string& name): LayerDecorator("ArgMax", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  Builder::ArgMaxLayer::ArgMaxLayer(const std::string& name): LayerDecorator('ArgMax', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
}

Builder::ArgMaxLayer::ArgMaxLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  Builder::ArgMaxLayer::ArgMaxLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ArgMax");
}

Builder::ArgMaxLayer::ArgMaxLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  Builder::ArgMaxLayer::ArgMaxLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ArgMax");
}

Builder::ArgMaxLayer& Builder::ArgMaxLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  Builder::ArgMaxLayer& Builder::ArgMaxLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::ArgMaxLayer::getPort() const {
    return getLayer()->getInputPorts()[0];
}

Builder::ArgMaxLayer& Builder::ArgMaxLayer::setPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  Builder::ArgMaxLayer& Builder::ArgMaxLayer::setPort(const Port &port) {" << std::endl;
    getLayer()->getInputPorts()[0] = port;
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

int Builder::ArgMaxLayer::getAxis() const {
    return getLayer()->getParameters().at("axis");
}
Builder::ArgMaxLayer& Builder::ArgMaxLayer::setAxis(int axis) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  Builder::ArgMaxLayer& Builder::ArgMaxLayer::setAxis(int axis) {" << std::endl;
    getLayer()->getParameters()["axis"] = axis;
    return *this;
}
size_t Builder::ArgMaxLayer::getTopK() const {
    return getLayer()->getParameters().at("top_k");
}
Builder::ArgMaxLayer& Builder::ArgMaxLayer::setTopK(size_t topK) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  Builder::ArgMaxLayer& Builder::ArgMaxLayer::setTopK(size_t topK) {" << std::endl;
    getLayer()->getParameters()["top_k"] = topK;
    return *this;
}
size_t Builder::ArgMaxLayer::getOutMaxVal() const {
    return getLayer()->getParameters().at("out_max_val");
}
Builder::ArgMaxLayer& Builder::ArgMaxLayer::setOutMaxVal(size_t outMaxVal) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  Builder::ArgMaxLayer& Builder::ArgMaxLayer::setOutMaxVal(size_t outMaxVal) {" << std::endl;
    getLayer()->getParameters()["out_max_val"] = outMaxVal;
    return *this;
}

REG_VALIDATOR_FOR(ArgMax, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  REG_VALIDATOR_FOR(ArgMax, [] (const InferenceEngine::Builder::Layer::CPtr& input_layer, bool partial) {" << std::endl;
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:          input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {" << std::endl;
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
    Builder::ArgMaxLayer layer(input_layer);
    if (layer.getAxis() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:      if (layer.getAxis() > 1) {" << std::endl;
        THROW_IE_EXCEPTION << "axis supports only 0 and 1 values.";
    }
    if (layer.getOutMaxVal() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:      if (layer.getOutMaxVal() > 1) {" << std::endl;
        THROW_IE_EXCEPTION << "OutMaxVal supports only 0 and 1 values.";
    }
});

REG_CONVERTER_FOR(ArgMax, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_argmax_layer.cpp:  REG_CONVERTER_FOR(ArgMax, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["axis"] = cnnLayer->GetParamAsInt("axis");
    layer.getParameters()["top_k"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("top_k"));
    layer.getParameters()["out_max_val"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("out_max_val"));
});


