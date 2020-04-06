#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_cnn_layer_builder.h>

#include <limits>
#include <set>
#include <sstream>

#include "blob_factory.hpp"
#include "ie_memcpy.h"

namespace InferenceEngine {

namespace Builder {

IE_SUPPRESS_DEPRECATED_START

ConverterRegister::ConverterRegister(const std::string& type,
                                     const std::function<void(const CNNLayerPtr&, Layer&)>& converter) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:                                       const std::function<void(const CNNLayerPtr&, Layer&)>& converter) {" << std::endl;
    if (getConvertersHolder().converters.find(type) == getConvertersHolder().converters.end())
        getConvertersHolder().converters[type] = converter;
}

ConvertersHolder& ConverterRegister::getConvertersHolder() {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:  ConvertersHolder& ConverterRegister::getConvertersHolder() {" << std::endl;
    static Builder::ConvertersHolder holder;
    return holder;
}

Layer builderFromCNNLayer(const CNNLayerPtr& cnnLayer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:  Layer builderFromCNNLayer(const CNNLayerPtr& cnnLayer) {" << std::endl;
    Builder::Layer layer(cnnLayer->type, cnnLayer->name);
    std::vector<Port> inputPorts;
    for (const auto& data : cnnLayer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:      for (const auto& data : cnnLayer->insData) {" << std::endl;
        auto lockedData = data.lock();
        if (!lockedData) continue;
        inputPorts.emplace_back(lockedData->getTensorDesc().getDims());
    }

    std::vector<Port> outputPorts;
    for (const auto& data : cnnLayer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:      for (const auto& data : cnnLayer->outData) {" << std::endl;
        outputPorts.emplace_back(data->getTensorDesc().getDims());
    }

    size_t inputsCount = inputPorts.size();
    std::map<std::string, Blob::Ptr> blobs = cnnLayer->blobs;
    if (blobs.find("weights") != blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:      if (blobs.find('weights') != blobs.end()) {" << std::endl;
        auto port = Port();
        port.setParameter("type", "weights");
        inputPorts.push_back(port);
    }
    if (blobs.find("biases") != blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:      if (blobs.find('biases') != blobs.end()) {" << std::endl;
        if (inputsCount == inputPorts.size()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:          if (inputsCount == inputPorts.size()) {" << std::endl;
            auto port = Port();
            port.setParameter("type", "weights");
            inputPorts.push_back(port);
        }

        auto port = Port();
        port.setParameter("type", "biases");
        inputPorts.push_back(port);
    }
    for (const auto& it : blobs) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:      for (const auto& it : blobs) {" << std::endl;
        if (it.first == "weights" || it.first == "biases") continue;
        auto port = Port();
        port.setParameter("type", it.first);
        inputPorts.emplace_back(port);
    }

    std::map<std::string, Parameter> params;
    for (const auto& it : cnnLayer->params) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:      for (const auto& it : cnnLayer->params) {" << std::endl;
        params[it.first] = it.second;
    }

    layer.setInputPorts(inputPorts).setOutputPorts(outputPorts).setParameters(params);

    Builder::ConverterRegister::convert(cnnLayer, layer);

    return layer;
}

std::map<std::string, std::string> convertParameters2Strings(const std::map<std::string, Parameter>& parameters) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:  std::map<std::string, std::string> convertParameters2Strings(const std::map<std::string, Parameter>& parameters) {" << std::endl;
    std::map<std::string, std::string> oldParams;
    for (const auto& param : parameters) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:      for (const auto& param : parameters) {" << std::endl;
        // skip blobs and ports
        if (param.second.is<Blob::CPtr>() || param.second.is<Blob::Ptr>() || param.second.is<std::vector<Port>>() ||
            param.second.is<PreProcessInfo>())
            continue;
        if (param.second.is<std::string>() || param.second.is<std::vector<std::string>>()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:          if (param.second.is<std::string>() || param.second.is<std::vector<std::string>>()) {" << std::endl;
            oldParams[param.first] = Builder::convertParameter2String<std::string>(param.second);
        } else if (param.second.is<int>() || param.second.is<std::vector<int>>()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:          } else if (param.second.is<int>() || param.second.is<std::vector<int>>()) {" << std::endl;
            oldParams[param.first] = Builder::convertParameter2String<int>(param.second);
        } else if (param.second.is<float>() || param.second.is<std::vector<float>>()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:          } else if (param.second.is<float>() || param.second.is<std::vector<float>>()) {" << std::endl;
            oldParams[param.first] = Builder::convertParameter2String<float>(param.second);
        } else if (param.second.is<unsigned int>() || param.second.is<std::vector<unsigned int>>()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:          } else if (param.second.is<unsigned int>() || param.second.is<std::vector<unsigned int>>()) {" << std::endl;
            oldParams[param.first] = Builder::convertParameter2String<unsigned int>(param.second);
        } else if (param.second.is<size_t>() || param.second.is<std::vector<size_t>>()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:          } else if (param.second.is<size_t>() || param.second.is<std::vector<size_t>>()) {" << std::endl;
            oldParams[param.first] = Builder::convertParameter2String<size_t>(param.second);
        } else if (param.second.is<bool>() || param.second.is<std::vector<bool>>()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_layer_builder.cpp:          } else if (param.second.is<bool>() || param.second.is<std::vector<bool>>()) {" << std::endl;
            oldParams[param.first] = Builder::convertParameter2String<bool>(param.second);
        } else {
            THROW_IE_EXCEPTION << "Parameter " << param.first << " has unsupported parameter type!";
        }
    }
    return oldParams;
}

IE_SUPPRESS_DEPRECATED_END

}  // namespace Builder
}  // namespace InferenceEngine
