#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transformation_context.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

TransformationContext::TransformationContext(ICNNNetwork& network)
    : network(network), layers(CNNNetSortTopologically(network)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformation_context.cpp:      : network(network), layers(CNNNetSortTopologically(network)) {" << std::endl;
    auto it = details::CNNNetworkIterator(&network);
    auto end = details::CNNNetworkIterator();
    while (it != end) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformation_context.cpp:      while (it != end) {" << std::endl;
        _original_precisions_map[(*it)->name] = {};
        for (auto data : (*it)->outData) _original_precisions_map[(*it)->name][data->getName()] = data->getPrecision();
        it++;
    }
}

void TransformationContext::removeLayer(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformation_context.cpp:  void TransformationContext::removeLayer(const CNNLayer& layer) {" << std::endl;
    for (size_t i = 0lu; i < layers.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformation_context.cpp:      for (size_t i = 0lu; i < layers.size(); ++i) {" << std::endl;
        if ((layers[i] != nullptr) && (layers[i]->name == layer.name)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformation_context.cpp:          if ((layers[i] != nullptr) && (layers[i]->name == layer.name)) {" << std::endl;
            layers[i] = nullptr;
            break;
        }
    }
}
