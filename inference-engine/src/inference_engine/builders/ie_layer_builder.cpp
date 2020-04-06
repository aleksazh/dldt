#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_layer_builder.hpp>
#include <details/caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace InferenceEngine;

Builder::Layer::Layer(const std::string& type, const std::string& name)
    : id((std::numeric_limits<idx_t>::max)()), type(type), name(name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:      : id((std::numeric_limits<idx_t>::max)()), type(type), name(name) {" << std::endl;}

Builder::Layer::Layer(const ILayer::CPtr& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  Builder::Layer::Layer(const ILayer::CPtr& layer) {" << std::endl;
    id = layer->getId();
    name = layer->getName();
    type = layer->getType();
    inPorts = layer->getInputPorts();
    outPorts = layer->getOutputPorts();
    params = layer->getParameters();
}

Builder::Layer::Layer(idx_t id, const Builder::Layer& layer): Layer(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  Builder::Layer::Layer(idx_t id, const Builder::Layer& layer): Layer(layer) {" << std::endl;
    this->id = id;
}

idx_t Builder::Layer::getId() const noexcept {
    return id;
}

const std::string& Builder::Layer::getType() const noexcept {
    return type;
}
Builder::Layer& Builder::Layer::setType(const std::string& type) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  Builder::Layer& Builder::Layer::setType(const std::string& type) {" << std::endl;
    this->type = type;
    return *this;
}

const std::string& Builder::Layer::getName() const noexcept {
    return name;
}
Builder::Layer& Builder::Layer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  Builder::Layer& Builder::Layer::setName(const std::string& name) {" << std::endl;
    this->name = name;
    return *this;
}

const std::map<std::string, Parameter>& Builder::Layer::getParameters() const noexcept {
    return params;
}
std::map<std::string, Parameter>& Builder::Layer::getParameters() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  std::map<std::string, Parameter>& Builder::Layer::getParameters() {" << std::endl;
    return params;
}
Builder::Layer& Builder::Layer::setParameters(const std::map<std::string, Parameter>& params) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  Builder::Layer& Builder::Layer::setParameters(const std::map<std::string, Parameter>& params) {" << std::endl;
    getParameters() = params;
    return *this;
}

std::vector<Port>& Builder::Layer::getInputPorts() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  std::vector<Port>& Builder::Layer::getInputPorts() {" << std::endl;
    return inPorts;
}
const std::vector<Port>& Builder::Layer::getInputPorts() const noexcept {
    return inPorts;
}
Builder::Layer& Builder::Layer::setInputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  Builder::Layer& Builder::Layer::setInputPorts(const std::vector<Port>& ports) {" << std::endl;
    getInputPorts() = ports;
    return *this;
}

std::vector<Port>& Builder::Layer::getOutputPorts() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  std::vector<Port>& Builder::Layer::getOutputPorts() {" << std::endl;
    return outPorts;
}
const std::vector<Port>& Builder::Layer::getOutputPorts() const noexcept {
    return outPorts;
}
Builder::Layer& Builder::Layer::setOutputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  Builder::Layer& Builder::Layer::setOutputPorts(const std::vector<Port>& ports) {" << std::endl;
    getOutputPorts() = ports;
    return *this;
}

const ILayer::CPtr Builder::Layer::build() const {
    validate(true);
    return std::static_pointer_cast<const ILayer>(shared_from_this());
}

void Builder::Layer::addValidator(const std::string& type,
                                  const std::function<void(const Layer::CPtr&, bool)>& validator) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:                                    const std::function<void(const Layer::CPtr&, bool)>& validator) {" << std::endl;
    auto holder = getValidatorsHolder();
    if (holder->validators.find(type) == holder->validators.end()) holder->validators[type] = validator;
}

void Builder::Layer::validate(bool partial) const {
    if (getValidatorsHolder()->validators.find(type) != getValidatorsHolder()->validators.end())
        getValidatorsHolder()->validators[type](shared_from_this(), partial);
}

std::shared_ptr<Builder::ValidatorsHolder> Builder::Layer::getValidatorsHolder() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:  std::shared_ptr<Builder::ValidatorsHolder> Builder::Layer::getValidatorsHolder() {" << std::endl;
    static std::shared_ptr<ValidatorsHolder> localHolder;
    if (localHolder == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_builder.cpp:      if (localHolder == nullptr) {" << std::endl;
        localHolder = std::make_shared<ValidatorsHolder>();
    }
    return localHolder;
}
