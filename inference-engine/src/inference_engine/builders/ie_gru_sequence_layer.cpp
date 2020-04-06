#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_gru_sequence_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::GRUSequenceLayer::GRUSequenceLayer(const std::string& name): LayerDecorator("GRUSequence", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer::GRUSequenceLayer(const std::string& name): LayerDecorator('GRUSequence', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(2);
    getLayer()->getInputPorts().resize(5);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getInputPorts()[2].setParameter("type", "biases");
    getLayer()->getInputPorts()[3].setParameter("type", "optional");
}

Builder::GRUSequenceLayer::GRUSequenceLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer::GRUSequenceLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("GRUSequence");
}

Builder::GRUSequenceLayer::GRUSequenceLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer::GRUSequenceLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("GRUSequence");
}

Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const std::vector<Port>& Builder::GRUSequenceLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setInputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setInputPorts(const std::vector<Port>& ports) {" << std::endl;
    getLayer()->getInputPorts() = ports;
    return *this;
}

const std::vector<Port>& Builder::GRUSequenceLayer::getOutputPorts() const {
    return getLayer()->getOutputPorts();
}

Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setOutputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setOutputPorts(const std::vector<Port>& ports) {" << std::endl;
    getLayer()->getOutputPorts() = ports;
    return *this;
}
int Builder::GRUSequenceLayer::getHiddenSize() const {
    return getLayer()->getParameters().at("hidden_size");
}
Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setHiddenSize(int size) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setHiddenSize(int size) {" << std::endl;
    getLayer()->getParameters()["hidden_size"] = size;
    return *this;
}
bool Builder::GRUSequenceLayer::getSequenceDim() const {
    return getLayer()->getParameters().at("sequence_dim");
}
Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setSqquenceDim(bool flag) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setSqquenceDim(bool flag) {" << std::endl;
    getLayer()->getParameters()["sequence_dim"] = flag;
    return *this;
}
const std::vector<std::string>& Builder::GRUSequenceLayer::getActivations() const {
    return getLayer()->getParameters().at("activations");
}
Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setActivations(const std::vector<std::string>& activations) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setActivations(const std::vector<std::string>& activations) {" << std::endl;
    getLayer()->getParameters()["activations"] = activations;
    return *this;
}
const std::vector<float>& Builder::GRUSequenceLayer::getActivationsAlpha() const {
    return getLayer()->getParameters().at("activations_alpha");
}
Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setActivationsAlpha(const std::vector<float>& activations) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setActivationsAlpha(const std::vector<float>& activations) {" << std::endl;
    getLayer()->getParameters()["activations_alpha"] = activations;
    return *this;
}
const std::vector<float>& Builder::GRUSequenceLayer::getActivationsBeta() const {
    return getLayer()->getParameters().at("activations_beta");
}
Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setActivationsBeta(const std::vector<float>& activations) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setActivationsBeta(const std::vector<float>& activations) {" << std::endl;
    getLayer()->getParameters()["activations_beta"] = activations;
    return *this;
}
float Builder::GRUSequenceLayer::getClip() const {
    return getLayer()->getParameters().at("clip");
}
Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setClip(float clip) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setClip(float clip) {" << std::endl;
    getLayer()->getParameters()["clip"] = clip;
    return *this;
}

bool Builder::GRUSequenceLayer::getLinearBeforeReset() const {
    return getLayer()->getParameters().at("linear_before_reset");
}
Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setLinearBeforeReset(bool flag) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setLinearBeforeReset(bool flag) {" << std::endl;
    getLayer()->getParameters()["linear_before_reset"] = flag;
    return *this;
}
const std::string& Builder::GRUSequenceLayer::getDirection() const {
    return getLayer()->getParameters().at("direction");
}
Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setDirection(const std::string& direction) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  Builder::GRUSequenceLayer& Builder::GRUSequenceLayer::setDirection(const std::string& direction) {" << std::endl;
    getLayer()->getParameters()["direction"] = direction;
    return *this;
}

REG_CONVERTER_FOR(GRUSequence, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:  REG_CONVERTER_FOR(GRUSequence, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["hidden_size"] = cnnLayer->GetParamAsInt("hidden_size");
    layer.getParameters()["sequence_dim"] = cnnLayer->GetParamAsBool("sequence_dim", true);
    std::vector<std::string> activations;
    std::istringstream stream(cnnLayer->GetParamAsString("activations"));
    std::string str;
    while (getline(stream, str, ',')) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_gru_sequence_layer.cpp:      while (getline(stream, str, ',')) {" << std::endl;
         activations.push_back(str);
    }
    layer.getParameters()["activations"] = activations;
    layer.getParameters()["activations_alpha"] = cnnLayer->GetParamAsFloats("activations_alpha");
    layer.getParameters()["activations_beta"] = cnnLayer->GetParamAsFloats("activations_beta");
    layer.getParameters()["clip"] = cnnLayer->GetParamAsFloat("clip");
    layer.getParameters()["linear_before_reset"] = cnnLayer->GetParamAsBool("linear_before_reset", true);
    layer.getParameters()["direction"] = cnnLayer->GetParamAsString("direction", "");
});


