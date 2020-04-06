#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_context.hpp>
#include <memory>
#include <shape_infer/built-in/ie_built_in_holder.hpp>
#include <string>
#include <vector>

using namespace InferenceEngine;

Context::Context() {
    std::cerr << "./inference-engine/src/inference_engine/ie_context.cpp:  Context::Context() {" << std::endl;
    auto builtIn = std::make_shared<ShapeInfer::BuiltInShapeInferHolder>();
    try {
        addExtension(builtIn);
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/ie_context.cpp:      } catch (...) {" << std::endl;
    }
}

void Context::addExtension(const IShapeInferExtensionPtr& ext) {
    std::cerr << "./inference-engine/src/inference_engine/ie_context.cpp:  void Context::addExtension(const IShapeInferExtensionPtr& ext) {" << std::endl;
    // Get all shape infer impls
    char** types = nullptr;
    unsigned int size = 0;
    ResponseDesc resp;
    StatusCode sts = ext->getShapeInferTypes(types, size, &resp);
    if (sts != OK) THROW_IE_EXCEPTION << "Failed to get types from extension: " << resp.msg;
    std::vector<std::string> implTypes;

    std::string badLayerTypes;
    for (int i = 0; i < size; i++) {
    std::cerr << "./inference-engine/src/inference_engine/ie_context.cpp:      for (int i = 0; i < size; i++) {" << std::endl;
        std::string type(types[i], strlen(types[i]));
        delete[] types[i];
        if (shapeInferImpls.find(type) != shapeInferImpls.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_context.cpp:          if (shapeInferImpls.find(type) != shapeInferImpls.end()) {" << std::endl;
            if (!badLayerTypes.empty()) badLayerTypes += ", ";
            badLayerTypes += type;
        }
        implTypes.emplace_back(type);
    }
    delete[] types;

    if (!badLayerTypes.empty())
        THROW_IE_EXCEPTION << "Failed to add extension with already registered types: " << badLayerTypes;

    for (const auto& implType : implTypes) {
    std::cerr << "./inference-engine/src/inference_engine/ie_context.cpp:      for (const auto& implType : implTypes) {" << std::endl;
        IShapeInferImpl::Ptr impl;
        sts = ext->getShapeInferImpl(impl, implType.c_str(), &resp);
        if (sts != OK) THROW_IE_EXCEPTION << "Failed to get implementation for " << implType << "type: " << resp.msg;
        shapeInferImpls[implType] = impl;
    }
}

void Context::addShapeInferImpl(const std::string& type, const IShapeInferImpl::Ptr& impl) {
    std::cerr << "./inference-engine/src/inference_engine/ie_context.cpp:  void Context::addShapeInferImpl(const std::string& type, const IShapeInferImpl::Ptr& impl) {" << std::endl;
    if (shapeInferImpls.find(type) != shapeInferImpls.end())
        THROW_IE_EXCEPTION << "Failed to add implementation for already registered type: " << type;
    shapeInferImpls[type] = impl;
}

IShapeInferImpl::Ptr Context::getShapeInferImpl(const std::string& type) {
    std::cerr << "./inference-engine/src/inference_engine/ie_context.cpp:  IShapeInferImpl::Ptr Context::getShapeInferImpl(const std::string& type) {" << std::endl;
    return shapeInferImpls.find(type) == shapeInferImpls.end() ? nullptr : shapeInferImpls[type];
}
