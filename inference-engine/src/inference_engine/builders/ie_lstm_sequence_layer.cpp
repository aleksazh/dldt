#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_lstm_sequence_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::LSTMSequenceLayer::LSTMSequenceLayer(const std::string& name): LayerDecorator("LSTMSequence", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer::LSTMSequenceLayer(const std::string& name): LayerDecorator('LSTMSequence', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(3);
    getLayer()->getInputPorts().resize(7);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getInputPorts()[2].setParameter("type", "biases");
    getLayer()->getInputPorts()[3].setParameter("type", "optional");
    getLayer()->getInputPorts()[6].setParameter("type", "weights");
}

Builder::LSTMSequenceLayer::LSTMSequenceLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer::LSTMSequenceLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("LSTMSequence");
}

Builder::LSTMSequenceLayer::LSTMSequenceLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer::LSTMSequenceLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("LSTMSequence");
}

Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const std::vector<Port>& Builder::LSTMSequenceLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setInputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setInputPorts(const std::vector<Port>& ports) {" << std::endl;
    getLayer()->getInputPorts() = ports;
    return *this;
}

const std::vector<Port>& Builder::LSTMSequenceLayer::getOutputPorts() const {
    return getLayer()->getOutputPorts();
}

Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setOutputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setOutputPorts(const std::vector<Port>& ports) {" << std::endl;
    getLayer()->getOutputPorts() = ports;
    return *this;
}
int Builder::LSTMSequenceLayer::getHiddenSize() const {
    return getLayer()->getParameters().at("hidden_size");
}
Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setHiddenSize(int size) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setHiddenSize(int size) {" << std::endl;
    getLayer()->getParameters()["hidden_size"] = size;
    return *this;
}
bool Builder::LSTMSequenceLayer::getSequenceDim() const {
    return getLayer()->getParameters().at("sequence_dim");
}
Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setSqquenceDim(bool flag) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setSqquenceDim(bool flag) {" << std::endl;
    getLayer()->getParameters()["sequence_dim"] = flag;
    return *this;
}
const std::vector<std::string>& Builder::LSTMSequenceLayer::getActivations() const {
    return getLayer()->getParameters().at("activations");
}
Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setActivations(const std::vector<std::string>& activations) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setActivations(const std::vector<std::string>& activations) {" << std::endl;
    getLayer()->getParameters()["activations"] = activations;
    return *this;
}
const std::vector<float>& Builder::LSTMSequenceLayer::getActivationsAlpha() const {
    return getLayer()->getParameters().at("activations_alpha");
}
Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setActivationsAlpha(const std::vector<float>& activations) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setActivationsAlpha(const std::vector<float>& activations) {" << std::endl;
    getLayer()->getParameters()["activations_alpha"] = activations;
    return *this;
}
const std::vector<float>& Builder::LSTMSequenceLayer::getActivationsBeta() const {
    return getLayer()->getParameters().at("activations_beta");
}
Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setActivationsBeta(const std::vector<float>& activations) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setActivationsBeta(const std::vector<float>& activations) {" << std::endl;
    getLayer()->getParameters()["activations_beta"] = activations;
    return *this;
}
float Builder::LSTMSequenceLayer::getClip() const {
    return getLayer()->getParameters().at("clip");
}
Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setClip(float clip) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setClip(float clip) {" << std::endl;
    getLayer()->getParameters()["clip"] = clip;
    return *this;
}

bool Builder::LSTMSequenceLayer::getInputForget() const {
    return getLayer()->getParameters().at("input_forget");
}
Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setInputForget(bool flag) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setInputForget(bool flag) {" << std::endl;
    getLayer()->getParameters()["input_forget"] = flag;
    return *this;
}
const std::string& Builder::LSTMSequenceLayer::getDirection() const {
    return getLayer()->getParameters().at("direction");
}
Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setDirection(const std::string& direction) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  Builder::LSTMSequenceLayer& Builder::LSTMSequenceLayer::setDirection(const std::string& direction) {" << std::endl;
    getLayer()->getParameters()["direction"] = direction;
    return *this;
}

REG_CONVERTER_FOR(LSTMSequence, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:  REG_CONVERTER_FOR(LSTMSequence, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["hidden_size"] = cnnLayer->GetParamAsInt("hidden_size");
    layer.getParameters()["sequence_dim"] = cnnLayer->GetParamAsBool("sequence_dim", true);
    std::vector<std::string> activations;
    std::istringstream stream(cnnLayer->GetParamAsString("activations"));
    std::string str;
    while (getline(stream, str, ',')) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_lstm_sequence_layer.cpp:      while (getline(stream, str, ',')) {" << std::endl;
         activations.push_back(str);
    }
    layer.getParameters()["activations"] = activations;
    layer.getParameters()["activations_alpha"] = cnnLayer->GetParamAsFloats("activations_alpha");
    layer.getParameters()["activations_beta"] = cnnLayer->GetParamAsFloats("activations_beta");
    layer.getParameters()["clip"] = cnnLayer->GetParamAsFloat("clip");
    layer.getParameters()["input_forget"] = cnnLayer->GetParamAsBool("input_forget", true);
    layer.getParameters()["direction"] = cnnLayer->GetParamAsString("direction", "");
});


