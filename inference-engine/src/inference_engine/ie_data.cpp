#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_data.h"

#include <map>
#include <memory>
#include <string>

#include "blob_factory.hpp"
#include "ie_layers.h"

using namespace InferenceEngine;

Blob::Ptr Blob::CreateFromData(const DataPtr& data) {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:  Blob::Ptr Blob::CreateFromData(const DataPtr& data) {" << std::endl;
    return CreateBlobFromData(data);
}

Data::Data(const std::string& name, Precision _precision, Layout layout)
    : name(name), userObject({0}), tensorDesc(_precision, layout) {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:      : name(name), userObject({0}), tensorDesc(_precision, layout) {" << std::endl;}

Data::Data(const std::string& name, const SizeVector& a_dims, Precision _precision, Layout layout)
    : name(name), userObject({0}), tensorDesc(_precision, SizeVector(a_dims.rbegin(), a_dims.rend()), layout) {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:      : name(name), userObject({0}), tensorDesc(_precision, SizeVector(a_dims.rbegin(), a_dims.rend()), layout) {" << std::endl;}

Data::Data(const std::string& name, const TensorDesc& desc): name(name), userObject({0}), tensorDesc(desc) {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:  Data::Data(const std::string& name, const TensorDesc& desc): name(name), userObject({0}), tensorDesc(desc) {" << std::endl;}

const Precision& Data::getPrecision() const {
    return tensorDesc.getPrecision();
}

const TensorDesc& Data::getTensorDesc() const {
    return tensorDesc;
}

bool Data::isInitialized() const {
    return !tensorDesc.getDims().empty() || tensorDesc.getLayout() == SCALAR;
}

void Data::setDims(const SizeVector& a_dims) {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:  void Data::setDims(const SizeVector& a_dims) {" << std::endl;
    tensorDesc.setDims(a_dims);
}

void Data::setLayout(Layout layout) {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:  void Data::setLayout(Layout layout) {" << std::endl;
    tensorDesc.setLayout(layout);
}

void Data::reshape(const SizeVector& a_dims, Layout a_layout) {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:  void Data::reshape(const SizeVector& a_dims, Layout a_layout) {" << std::endl;
    tensorDesc.reshape(a_dims, a_layout);
}

CNNLayerWeakPtr& Data::getCreatorLayer() {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:  CNNLayerWeakPtr& Data::getCreatorLayer() {" << std::endl;
    return creatorLayer;
}

const std::string& Data::getName() const {
    return name;
}

void Data::setName(const std::string& newName) {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:  void Data::setName(const std::string& newName) {" << std::endl;
    name = newName;
}

std::map<std::string, CNNLayerPtr>& Data::getInputTo() {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:  std::map<std::string, CNNLayerPtr>& Data::getInputTo() {" << std::endl;
    return inputTo;
}

const UserValue& Data::getUserObject() const {
    return userObject;
}

Layout Data::getLayout() const {
    return tensorDesc.getLayout();
}

void Data::setPrecision(const Precision& precision) {
    std::cerr << "./inference-engine/src/inference_engine/ie_data.cpp:  void Data::setPrecision(const Precision& precision) {" << std::endl;
    tensorDesc.setPrecision(precision);
}

const SizeVector& Data::getDims() const {
    return tensorDesc.getDims();
}
