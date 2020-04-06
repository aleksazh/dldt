#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_rnn_sequence_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::RNNSequenceLayer::RNNSequenceLayer(const std::string& name): LayerDecorator("RNNSequence", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer::RNNSequenceLayer(const std::string& name): LayerDecorator('RNNSequence', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(2);
    getLayer()->getInputPorts().resize(5);
    getLayer()->getInputPorts()[1].setParameter("type", "weights");
    getLayer()->getInputPorts()[2].setParameter("type", "biases");
    getLayer()->getInputPorts()[3].setParameter("type", "optional");
}

Builder::RNNSequenceLayer::RNNSequenceLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer::RNNSequenceLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("RNNSequence");
}

Builder::RNNSequenceLayer::RNNSequenceLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer::RNNSequenceLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("RNNSequence");
}

Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const std::vector<Port>& Builder::RNNSequenceLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setInputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setInputPorts(const std::vector<Port>& ports) {" << std::endl;
    getLayer()->getInputPorts() = ports;
    return *this;
}

const std::vector<Port>& Builder::RNNSequenceLayer::getOutputPorts() const {
    return getLayer()->getOutputPorts();
}

Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setOutputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setOutputPorts(const std::vector<Port>& ports) {" << std::endl;
    getLayer()->getOutputPorts() = ports;
    return *this;
}
int Builder::RNNSequenceLayer::getHiddenSize() const {
    return getLayer()->getParameters().at("hidden_size");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setHiddenSize(int size) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setHiddenSize(int size) {" << std::endl;
    getLayer()->getParameters()["hidden_size"] = size;
    return *this;
}
bool Builder::RNNSequenceLayer::getSequenceDim() const {
    return getLayer()->getParameters().at("sequence_dim");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setSqquenceDim(bool flag) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setSqquenceDim(bool flag) {" << std::endl;
    getLayer()->getParameters()["sequence_dim"] = flag;
    return *this;
}
const std::vector<std::string>& Builder::RNNSequenceLayer::getActivations() const {
    return getLayer()->getParameters().at("activations");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setActivations(const std::vector<std::string>& activations) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setActivations(const std::vector<std::string>& activations) {" << std::endl;
    getLayer()->getParameters()["activations"] = activations;
    return *this;
}
const std::vector<float>& Builder::RNNSequenceLayer::getActivationsAlpha() const {
    return getLayer()->getParameters().at("activations_alpha");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setActivationsAlpha(const std::vector<float>& activations) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setActivationsAlpha(const std::vector<float>& activations) {" << std::endl;
    getLayer()->getParameters()["activations_alpha"] = activations;
    return *this;
}
const std::vector<float>& Builder::RNNSequenceLayer::getActivationsBeta() const {
    return getLayer()->getParameters().at("activations_beta");
}
Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setActivationsBeta(const std::vector<float>& activations) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  Builder::RNNSequenceLayer& Builder::RNNSequenceLayer::setActivationsBeta(const std::vector<float>& activations) {" << std::endl;
    getLayer()->getParameters()["activations_beta"] = activations;
    return *this;
}
REG_CONVERTER_FOR(RNNSequence, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:  REG_CONVERTER_FOR(RNNSequence, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["hidden_size"] = cnnLayer->GetParamAsInt("hidden_size");
    layer.getParameters()["sequence_dim"] = cnnLayer->GetParamAsBool("sequence_dim", true);
    std::vector<std::string> activations;
    std::istringstream stream(cnnLayer->GetParamAsString("activations"));
    std::string str;
    while (getline(stream, str, ',')) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_rnn_sequence_layer.cpp:      while (getline(stream, str, ',')) {" << std::endl;
         activations.push_back(str);
    }
    layer.getParameters()["activations"] = activations;
    layer.getParameters()["activations_alpha"] = cnnLayer->GetParamAsFloats("activations_alpha");
    layer.getParameters()["activations_beta"] = cnnLayer->GetParamAsFloats("activations_beta");
});


