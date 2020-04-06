#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_network.hpp>
#include <map>
#include <memory>
#include <string>

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

PortData::PortData() {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  PortData::PortData() {" << std::endl;
    createData({});
}

PortData::PortData(const SizeVector& shape, const Precision& precision) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  PortData::PortData(const SizeVector& shape, const Precision& precision) {" << std::endl;
    createData({precision, shape, TensorDesc::getLayoutByDims(shape)});
}

const Blob::Ptr& PortData::getData() const {
    return data;
}

void PortData::setData(const Blob::Ptr& data) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  void PortData::setData(const Blob::Ptr& data) {" << std::endl;
    this->data = data;
}

const std::map<std::string, Parameter>& PortData::getParameters() const noexcept {
    return parameters;
}

void PortData::createData(const TensorDesc& desc) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  void PortData::createData(const TensorDesc& desc) {" << std::endl;
    switch (desc.getPrecision()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:      switch (desc.getPrecision()) {" << std::endl;
    case Precision::UNSPECIFIED:
        data = std::make_shared<InferenceEngine::TBlob<uint8_t>>(desc);
        break;
    case Precision::FP32:
        data = make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(desc);
        break;
    case Precision::FP16:
        data = make_shared_blob<PrecisionTrait<Precision::FP16>::value_type>(desc);
        break;
    case Precision::Q78:
        data = make_shared_blob<PrecisionTrait<Precision::Q78>::value_type>(desc);
        break;
    case Precision::I16:
        data = make_shared_blob<PrecisionTrait<Precision::I16>::value_type>(desc);
        break;
    case Precision::U8:
        data = make_shared_blob<PrecisionTrait<Precision::U8>::value_type>(desc);
        break;
    case Precision::I8:
        data = make_shared_blob<PrecisionTrait<Precision::I8>::value_type>(desc);
        break;
    case Precision::U16:
        data = make_shared_blob<PrecisionTrait<Precision::U16>::value_type>(desc);
        break;
    case Precision::I32:
        data = make_shared_blob<PrecisionTrait<Precision::I32>::value_type>(desc);
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported precisions!";
    }
}

void PortData::setShape(const SizeVector& shape) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  void PortData::setShape(const SizeVector& shape) {" << std::endl;
    TensorDesc desc = data->getTensorDesc();
    if (desc.getDims() == shape) return;
    if (data->cbuffer() != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:      if (data->cbuffer() != nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "Cannot change shape for allocated data!";
    }
    createData({desc.getPrecision(), shape, TensorDesc::getLayoutByDims(shape)});
}

Port::Port() {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  Port::Port() {" << std::endl;
    data = std::make_shared<PortData>();
}

Port::Port(const SizeVector& shapes, const Precision& precision) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  Port::Port(const SizeVector& shapes, const Precision& precision) {" << std::endl;
    data = std::make_shared<PortData>(shapes, precision);
}
Port::Port(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  Port::Port(const Port& port) {" << std::endl;
    parameters = port.parameters;
    data = port.data;
}

bool Port::operator==(const Port& rhs) const {
    return parameters == rhs.parameters && data == rhs.data;
}

bool Port::operator!=(const Port& rhs) const {
    return !(rhs == *this);
}

const SizeVector& Port::shape() const noexcept {
    return data->getData()->getTensorDesc().getDims();
}

void Port::setShape(const SizeVector& shape) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  void Port::setShape(const SizeVector& shape) {" << std::endl;
    data->setShape(shape);
}

const std::map<std::string, Parameter>& Port::getParameters() const noexcept {
    return parameters;
}

void Port::setParameters(const std::map<std::string, Parameter>& params) noexcept {
    parameters = params;
}

void Port::setParameter(const std::string& name, const Parameter& param) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  void Port::setParameter(const std::string& name, const Parameter& param) {" << std::endl;
    parameters[name] = param;
}

const PortData::Ptr& Port::getData() const noexcept {
    return data;
}

void Port::setData(const PortData::Ptr& data) {
    std::cerr << "./inference-engine/src/inference_engine/ie_network.cpp:  void Port::setData(const PortData::Ptr& data) {" << std::endl;
    if (!data) return;
    this->data = data;
}

IE_SUPPRESS_DEPRECATED_END
