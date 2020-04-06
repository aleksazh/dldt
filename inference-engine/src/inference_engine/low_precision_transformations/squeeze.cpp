#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/squeeze.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <string>
#include <vector>


using namespace InferenceEngine;
using namespace InferenceEngine::details;

void SqueezeTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:      if (!canBeTransformed(context, layer)) {" << std::endl;
        return;
    }

    if ((layer.insData.size() == 0) || (layer.insData.size() > 2)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:      if ((layer.insData.size() == 0) || (layer.insData.size() > 2)) {" << std::endl;
        THROW_IE_EXCEPTION << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Squeeze")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:      if (!CaselessEq<std::string>()(layer.type, 'Squeeze')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer '" << layer.name << "' is not correct";
    }

    if (layer.insData.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:      if (layer.insData.size() > 1) {" << std::endl;
        CNNLayerPtr constLayer = CNNNetworkHelper::getParent(layer, 1);
        if ((constLayer != nullptr) && (constLayer->type != "Const")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          if ((constLayer != nullptr) && (constLayer->type != 'Const')) {" << std::endl;
            return;
        }

        Blob::Ptr paramsBlob = CNNNetworkHelper::getBlob(constLayer, "custom");
        if (paramsBlob->getTensorDesc().getPrecision() != Precision::I32) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          if (paramsBlob->getTensorDesc().getPrecision() != Precision::I32) {" << std::endl;
            THROW_IE_EXCEPTION << "unexpected precision " << paramsBlob->getTensorDesc().getPrecision();
        }

        DataPtr inputData = layer.insData[0].lock();
        if (inputData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          if (inputData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input data is absent";
        }

        const std::vector<size_t> inputDims = inputData->getTensorDesc().getDims();
        if (inputDims.size() < paramsBlob->size()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          if (inputDims.size() < paramsBlob->size()) {" << std::endl;
            return;
        }

        const signed int* paramsBuffer = paramsBlob->buffer().as<const signed int*>();
        for (size_t index = 0; index < paramsBlob->size(); ++index) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          for (size_t index = 0; index < paramsBlob->size(); ++index) {" << std::endl;
            if ((paramsBuffer[index] == 0) || (paramsBuffer[index] == 1)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:              if ((paramsBuffer[index] == 0) || (paramsBuffer[index] == 1)) {" << std::endl;
                return;
            }
        }
    } else {
        if (layer.outData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          if (layer.outData.size() != 1) {" << std::endl;
            THROW_IE_EXCEPTION << "unexpected output count " << layer.outData.size();
        }
        const std::vector<size_t> outputDims = layer.outData[0]->getDims();

        auto it = std::find(outputDims.begin(), outputDims.end(), 1lu);
        if (it != outputDims.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          if (it != outputDims.end()) {" << std::endl;
            return;
        }

        if (layer.insData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          if (layer.insData.size() != 1) {" << std::endl;
            THROW_IE_EXCEPTION << "unexpected input count " << layer.insData.size();
        }
        const DataPtr input = layer.insData[0].lock();
        if (input == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          if (input == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input is absent";
        }
        const std::vector<size_t> inputDims = input->getDims();
        for (size_t i = 0; (i < 2) && (i < outputDims.size()); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:          for (size_t i = 0; (i < 2) && (i < outputDims.size()); ++i) {" << std::endl;
            if (inputDims[i] != outputDims[i]) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/squeeze.cpp:              if (inputDims[i] != outputDims[i]) {" << std::endl;
                return;
            }
        }
    }

    TransparentBaseTransformation::transform(context, layer);
}
