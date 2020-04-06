#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_batch_normalization_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <string>

using namespace InferenceEngine;

Builder::BatchNormalizationLayer::BatchNormalizationLayer(const std::string& name): LayerDecorator("BatchNormalization", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp:  Builder::BatchNormalizationLayer::BatchNormalizationLayer(const std::string& name): LayerDecorator('BatchNormalization', name) {" << std::endl;
    getLayer()->getInputPorts().resize(3);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getInputPorts()[2].setParameter("type", "biases");
    getLayer()->getOutputPorts().resize(1);
    setEpsilon(0.00000001f);
}

Builder::BatchNormalizationLayer::BatchNormalizationLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp:  Builder::BatchNormalizationLayer::BatchNormalizationLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("BatchNormalization");
}

Builder::BatchNormalizationLayer::BatchNormalizationLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp:  Builder::BatchNormalizationLayer::BatchNormalizationLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("BatchNormalization");
}

Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp:  Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const Port& Builder::BatchNormalizationLayer::getPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp:  Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    getLayer()->getInputPorts()[0] = port;
    return *this;
}

float Builder::BatchNormalizationLayer::getEpsilon() const {
    return getLayer()->getParameters().at("epsilon");
}
Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setEpsilon(float eps) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp:  Builder::BatchNormalizationLayer& Builder::BatchNormalizationLayer::setEpsilon(float eps) {" << std::endl;
    getLayer()->getParameters()["epsilon"] = eps;
    return *this;
}

REG_VALIDATOR_FOR(BatchNormalization, [](const Builder::Layer::CPtr& layer, bool partial)  {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp:  REG_VALIDATOR_FOR(BatchNormalization, [](const Builder::Layer::CPtr& layer, bool partial)  {" << std::endl;
    Builder::BatchNormalizationLayer batchNormBuilder(layer);
    if (partial)
        return;
    auto weights = layer->getInputPorts()[1].getData()->getData();
    auto biases = layer->getInputPorts()[2].getData()->getData();
    if (!weights || weights->cbuffer() == nullptr || !biases || biases->cbuffer() == nullptr)
        THROW_IE_EXCEPTION << "Cannot create BatchNormalization layer! Weights and biases are required!";
});

REG_CONVERTER_FOR(BatchNormalization, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_batch_normalization_layer.cpp:  REG_CONVERTER_FOR(BatchNormalization, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["epsilon"] = cnnLayer->GetParamAsFloat("epsilon");
});