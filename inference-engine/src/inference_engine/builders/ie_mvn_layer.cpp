#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_mvn_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::MVNLayer::MVNLayer(const std::string& name): LayerDecorator("MVN", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  Builder::MVNLayer::MVNLayer(const std::string& name): LayerDecorator('MVN', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(1);
    setEpsilon(9.999999717180685e-10f);
    setNormalize(true);
    setAcrossChannels(true);
}

Builder::MVNLayer::MVNLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  Builder::MVNLayer::MVNLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("MVN");
}

Builder::MVNLayer::MVNLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  Builder::MVNLayer::MVNLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("MVN");
}

Builder::MVNLayer& Builder::MVNLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  Builder::MVNLayer& Builder::MVNLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::MVNLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::MVNLayer& Builder::MVNLayer::setPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  Builder::MVNLayer& Builder::MVNLayer::setPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

bool Builder::MVNLayer::getAcrossChannels() const {
    return getLayer()->getParameters().at("across_channels");
}
Builder::MVNLayer& Builder::MVNLayer::setAcrossChannels(bool flag) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  Builder::MVNLayer& Builder::MVNLayer::setAcrossChannels(bool flag) {" << std::endl;
    getLayer()->getParameters()["across_channels"] = flag ? 1 : 0;
    return *this;
}
bool Builder::MVNLayer::getNormalize() const {
    return getLayer()->getParameters().at("normalize_variance");
}
Builder::MVNLayer& Builder::MVNLayer::setNormalize(bool flag) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  Builder::MVNLayer& Builder::MVNLayer::setNormalize(bool flag) {" << std::endl;
    getLayer()->getParameters()["normalize_variance"] = flag ? 1 : 0;
    return *this;
}
float Builder::MVNLayer::getEpsilon() const {
    return getLayer()->getParameters().at("eps");
}
Builder::MVNLayer& Builder::MVNLayer::setEpsilon(float eps) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  Builder::MVNLayer& Builder::MVNLayer::setEpsilon(float eps) {" << std::endl;
    getLayer()->getParameters()["eps"] = eps;
    return *this;
}

REG_VALIDATOR_FOR(MVN, [](const Builder::Layer::CPtr& input_layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  REG_VALIDATOR_FOR(MVN, [](const Builder::Layer::CPtr& input_layer, bool partial) {" << std::endl;
    Builder::MVNLayer layer(input_layer);
    if (layer.getEpsilon() <= 0) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:      if (layer.getEpsilon() <= 0) {" << std::endl;
        THROW_IE_EXCEPTION << "Epsilon should be > 0";
    }
    if (!input_layer->getInputPorts().empty() &&
        !input_layer->getOutputPorts().empty() &&
        !input_layer->getInputPorts()[0].shape().empty() &&
        !input_layer->getOutputPorts()[0].shape().empty() &&
        input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:          input_layer->getInputPorts()[0].shape() != input_layer->getOutputPorts()[0].shape()) {" << std::endl;
        THROW_IE_EXCEPTION << "Input and output ports should be equal";
    }
});

REG_CONVERTER_FOR(MVN, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_mvn_layer.cpp:  REG_CONVERTER_FOR(MVN, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["across_channels"] = cnnLayer->GetParamAsBool("across_channels", 0);
    layer.getParameters()["normalize_variance"] = cnnLayer->GetParamAsBool("normalize_variance", 0);
    layer.getParameters()["eps"] = cnnLayer->GetParamAsFloat("eps", 0);
});