#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transformer.hpp"

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <cmath>
#include <details/caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"
#include "network_helper.hpp"

#include "activation.hpp"
#include "concat_multi_channels.hpp"
#include "const.hpp"
#include "convolution.hpp"
#include "eltwise.hpp"
#include "fake_quantize.hpp"
#include "fully_connected.hpp"
#include "fuse_fake_quantize_and_scale_shift.hpp"
#include "mvn.hpp"
#include "permute.hpp"
#include "pooling.hpp"
#include "reshape.hpp"
#include "scaleshift_to_convolution.hpp"
#include "squeeze.hpp"

// uncomment to display precision info during low precision transformations
// #define DISPLAY_PECISION

using namespace InferenceEngine;
using namespace InferenceEngine::details;

LowPrecisionTransformations::LowPrecisionTransformations(
    const std::map<std::string, LayerTransformationPtr>& branchSpecificTransformations,
    const std::map<std::string, LayerTransformationPtr>& transformations,
    const std::map<std::string, LayerTransformationPtr>& cleanupTransformations) :
    branchSpecificTransformations(branchSpecificTransformations),
    transformations(transformations),
    cleanupTransformations(cleanupTransformations) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      cleanupTransformations(cleanupTransformations) {" << std::endl;}

void LowPrecisionTransformations::setUpdatePrecisions(const bool updatePrecisions) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:  void LowPrecisionTransformations::setUpdatePrecisions(const bool updatePrecisions) {" << std::endl;
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {" << std::endl;
        it->second->setUpdatePrecisions(updatePrecisions);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = transformations.begin(); it != transformations.end(); ++it) {" << std::endl;
        it->second->setUpdatePrecisions(updatePrecisions);
    }
}

void LowPrecisionTransformations::setQuantizeOutputs(const bool quantizeOutputs) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:  void LowPrecisionTransformations::setQuantizeOutputs(const bool quantizeOutputs) {" << std::endl;
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {" << std::endl;
        it->second->setQuantizeOutputs(quantizeOutputs);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = transformations.begin(); it != transformations.end(); ++it) {" << std::endl;
        it->second->setQuantizeOutputs(quantizeOutputs);
    }
}

void LowPrecisionTransformations::setWeightsToConst(const bool weightsToConst) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:  void LowPrecisionTransformations::setWeightsToConst(const bool weightsToConst) {" << std::endl;
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {" << std::endl;
        it->second->setWeightsToConst(weightsToConst);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = transformations.begin(); it != transformations.end(); ++it) {" << std::endl;
        it->second->setWeightsToConst(weightsToConst);
    }
}

void LowPrecisionTransformations::setQuantizedTensorAlignmentOnActivations(
    const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations) {" << std::endl;
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {" << std::endl;
        it->second->setQuantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = transformations.begin(); it != transformations.end(); ++it) {" << std::endl;
        it->second->setQuantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations);
    }
}

void LowPrecisionTransformations::setQuantizedTensorAlignmentOnWeights(
    const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights) {" << std::endl;
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {" << std::endl;
        it->second->setQuantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it = transformations.begin(); it != transformations.end(); ++it) {" << std::endl;
        it->second->setQuantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights);
    }
}

LowPrecisionTransformations& LowPrecisionTransformations::remove(const std::string& layerName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:  LowPrecisionTransformations& LowPrecisionTransformations::remove(const std::string& layerName) {" << std::endl;
    branchSpecificTransformations.erase(layerName);
    transformations.erase(layerName);
    return *this;
}

LayerTransformationPtr LowPrecisionTransformations::find(const std::string& layerType) const {
    auto it = branchSpecificTransformations.find(layerType);
    if (it != branchSpecificTransformations.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      if (it != branchSpecificTransformations.end()) {" << std::endl;
        return it->second;
    }

    it = transformations.find(layerType);
    if (it != transformations.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      if (it != transformations.end()) {" << std::endl;
        return it->second;
    }

    it = cleanupTransformations.find(layerType);
    if (it != cleanupTransformations.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      if (it != cleanupTransformations.end()) {" << std::endl;
        return it->second;
    }

    return nullptr;
}

void LowPrecisionTransformations::setParamsManager(IParamsManager* paramsManager) noexcept {
    setParamsManager(paramsManager, branchSpecificTransformations);
    setParamsManager(paramsManager, transformations);
    setParamsManager(paramsManager, cleanupTransformations);
}

void LowPrecisionTransformations::setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept {
    setLayerTransformationsManager(layerTransformationsManager, branchSpecificTransformations);
    setLayerTransformationsManager(layerTransformationsManager, transformations);
    setLayerTransformationsManager(layerTransformationsManager, cleanupTransformations);
}

void LowPrecisionTransformations::setParamsManager(
    IParamsManager* paramsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it : transformations) {" << std::endl;
        it.second->setParamsManager(paramsManager);
    }
}

void LowPrecisionTransformations::setLayerTransformationsManager(
    ILayerTransformationsManager* layerTransformationsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (auto it : transformations) {" << std::endl;
        it.second->setLayerTransformationsManager(layerTransformationsManager);
    }
}

LowPrecisionTransformations LowPrecisionTransformer::getAllTransformations(const LayerTransformation::Params& params) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:  LowPrecisionTransformations LowPrecisionTransformer::getAllTransformations(const LayerTransformation::Params& params) {" << std::endl;
    return LowPrecisionTransformations(
        std::map<std::string, LayerTransformationPtr>({
            { "Eltwise", LayerTransformationPtr(new EltwiseTransformation(params)) },
            { "Concat", LayerTransformationPtr(new ConcatMultiChannelsTransformation(params))}
        }),
        std::map<std::string, LayerTransformationPtr>({
            { "Convolution", LayerTransformationPtr(new ConvolutionTransformation(params)) },
            { "Pooling", LayerTransformationPtr(new PoolingTransformation(params)) },
            { "FakeQuantize", LayerTransformationPtr(new FakeQuantizeTransformation(params)) },
            { "Reshape", LayerTransformationPtr(new ReshapeTransformation(params)) },
            { "FullyConnected", LayerTransformationPtr(new FullyConnectedTransformation(params)) },
            { "GEMM", LayerTransformationPtr(new FullyConnectedTransformation(params)) },
            { "Permute", LayerTransformationPtr(new PermuteTransformation(params)) },
            { "Squeeze", LayerTransformationPtr(new SqueezeTransformation(params)) },
            { "ReLU", LayerTransformationPtr(new ActivationTransformation(params)) },
            { "MVN", LayerTransformationPtr(new MvnTransformation(params)) }
        }),
        std::map<std::string, LayerTransformationPtr>({
            { "FakeQuantize", LayerTransformationPtr(new FuseFakeQuantizeAndScaleShiftTransformation(params)) },
            { "ScaleShift", LayerTransformationPtr(new ScaleShiftToConvolutionTransformation(params)) },
        }));
}

LowPrecisionTransformer::LowPrecisionTransformer(): transformations(LowPrecisionTransformer::getAllTransformations()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:  LowPrecisionTransformer::LowPrecisionTransformer(): transformations(LowPrecisionTransformer::getAllTransformations()) {" << std::endl;}

LowPrecisionTransformer::LowPrecisionTransformer(const LowPrecisionTransformations& transformations)
    : transformations(transformations) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      : transformations(transformations) {" << std::endl;}

void LowPrecisionTransformer::renameLayersByType(const std::vector<CNNLayerPtr>& layers, const std::string& type) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:  void LowPrecisionTransformer::renameLayersByType(const std::vector<CNNLayerPtr>& layers, const std::string& type) {" << std::endl;
    size_t number = 1;
    for (size_t i = 0; i < layers.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (size_t i = 0; i < layers.size(); ++i) {" << std::endl;
        const CNNLayerPtr layer = layers[i];
        if (layer->type != type) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (layer->type != type) {" << std::endl;
            continue;
        }

        layer->name = layer->type + std::to_string(number);
        ++number;
    }
}

void LowPrecisionTransformer::rename(ICNNNetwork& network) const {
    TransformationContext context(network);

    const std::unordered_set<std::string> standaloneLayerTypes = {"Convolution", "Concat",  "Eltwise",
                                                                  "Reshape",     "Pooling", "Clamp"};
    for (const std::string& standaloneLayerType : standaloneLayerTypes) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (const std::string& standaloneLayerType : standaloneLayerTypes) {" << std::endl;
        renameLayersByType(context.getLayers(), standaloneLayerType);
    }

    size_t fakeQuantizeNumber = 1;
    for (size_t i = 0lu; i < context.getLayers().size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (size_t i = 0lu; i < context.getLayers().size(); ++i) {" << std::endl;
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer->type != "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (layer->type != 'FakeQuantize') {" << std::endl;
            continue;
        }

        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*layer);
        if ((children.size() == 1) && (children[0]->type == "Convolution")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if ((children.size() == 1) && (children[0]->type == 'Convolution')) {" << std::endl;
            const std::string postfix = CNNNetworkHelper::getIndex(*layer) == 0 ? "data" : "weights";
            layer->name = children[0]->name + "_FakeQuantize_" + postfix;
        } else {
            layer->name = layer->type + std::to_string(fakeQuantizeNumber);
            ++fakeQuantizeNumber;
        }
    }

    size_t otherNumber = 1;
    for (size_t i = 0; i < context.getLayers().size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (size_t i = 0; i < context.getLayers().size(); ++i) {" << std::endl;
        std::string name;
        const CNNLayerPtr layer = context.getLayers()[i];
        if ((standaloneLayerTypes.find(layer->type) != standaloneLayerTypes.end()) || (layer->type == "FakeQuantize")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if ((standaloneLayerTypes.find(layer->type) != standaloneLayerTypes.end()) || (layer->type == 'FakeQuantize')) {" << std::endl;
            continue;
        }

        if (layer->type == "Const") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (layer->type == 'Const') {" << std::endl;
            const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*layer);
            if (children.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:              if (children.size() == 1) {" << std::endl;
                if (children[0]->type == "Convolution") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:                  if (children[0]->type == 'Convolution') {" << std::endl;
                    const std::string postfix = CNNNetworkHelper::getIndex(*layer) == 1 ? "weights" : "biases";
                    name = children[0]->name + "_Const_" + postfix;
                } else if (children[0]->type == "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:                  } else if (children[0]->type == 'FakeQuantize') {" << std::endl;
                    name = children[0]->name + "_Const_" + std::to_string(CNNNetworkHelper::getIndex(*layer));
                }
            }
        }

        if (name.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (name.empty()) {" << std::endl;
            name = layer->type + std::to_string(otherNumber);
            ++otherNumber;
        }

        layer->name = name;
    }
}

void LowPrecisionTransformer::transform(ICNNNetwork& network) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:  void LowPrecisionTransformer::transform(ICNNNetwork& network) {" << std::endl;
    auto it = details::CNNNetworkIterator(&network);
    auto end = details::CNNNetworkIterator();
    bool fqFound = false;
    bool allFQareUnsupported = true;
    while (it != end) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      while (it != end) {" << std::endl;
        if (CaselessEq<std::string>()((*it)->type, "FakeQuantize")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (CaselessEq<std::string>()((*it)->type, 'FakeQuantize')) {" << std::endl;
            fqFound = true;
            if (QuantizationDetails::isSupportedLevel((*it)->GetParamAsUInt("levels"))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:              if (QuantizationDetails::isSupportedLevel((*it)->GetParamAsUInt('levels'))) {" << std::endl;
                allFQareUnsupported = false;
                break;
            }
        }
        it++;
    }
    // If network does not have FakeQuantize layers
    // or all found FQ layers are binary - do nothing and return
    if (!fqFound || allFQareUnsupported) return;

    transformations.setParamsManager(this);
    transformations.setLayerTransformationsManager(this);

    TransformationContext context(network);

    // TODO: branch specific transformations execution
    for (size_t i = 0lu; i < context.getLayers().size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (size_t i = 0lu; i < context.getLayers().size(); ++i) {" << std::endl;
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (layer == nullptr) {" << std::endl;
            continue;
        }

        const auto it = transformations.branchSpecificTransformations.find(layer->type);
        if (it == transformations.branchSpecificTransformations.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (it == transformations.branchSpecificTransformations.end()) {" << std::endl;
            continue;
        }
        it->second->transform(context, *layer);
    }

    // Step #1: FakeQuantize layer transformation execution
    LayerTransformationPtr fqTransformation = transformations.find("FakeQuantize");
    if (fqTransformation == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      if (fqTransformation == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "FakeQuantize transformation was not found";
    }
    for (size_t i = 0lu; i < context.getLayers().size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (size_t i = 0lu; i < context.getLayers().size(); ++i) {" << std::endl;
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (layer == nullptr) {" << std::endl;
            continue;
        }

        if (CaselessEq<std::string>()(layer->type, "FakeQuantize")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (CaselessEq<std::string>()(layer->type, 'FakeQuantize')) {" << std::endl;
            fqTransformation->transform(context, *layer);
        }
    }

    // Step #2: layer transformations execution
    for (size_t i = 0; i < context.getLayers().size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (size_t i = 0; i < context.getLayers().size(); ++i) {" << std::endl;
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (layer == nullptr) {" << std::endl;
            continue;
        }

        bool transformed;
        const auto it = transformations.transformations.find(layer->type);
        if (it != transformations.transformations.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (it != transformations.transformations.end()) {" << std::endl;
            it->second->transform(context, *layer);
            transformed = true;
        }

#ifdef DISPLAY_PECISION
        CNNLayerPtr transformedLayer = CNNNetworkHelper::getLayer(context.network, layer->name);
        if (transformedLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (transformedLayer == nullptr) {" << std::endl;
            if (layer->type == "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:              if (layer->type == 'FakeQuantize') {" << std::endl;
                std::cout << "Layer " << layer->name << ": " << QuantizationDetails::getDetails(*layer) << std::endl;
            }

            std::cout << "Layer was " << (transformed ? "transformed: " : "skipped: ") << layer->type << ", "
                      << layer->name << ": [REMOVED]" << std::endl;
        } else {
            if (transformedLayer->type == "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:              if (transformedLayer->type == 'FakeQuantize') {" << std::endl;
                std::cout << "Layer " << transformedLayer->name << ": "
                          << QuantizationDetails::getDetails(*transformedLayer) << std::endl;
            }

            std::cout << "Layer was " << (transformed ? "transformed: " : "skipped: ") << transformedLayer->type << ", "
                      << transformedLayer->name << ", output layer precision: "
                      << ((transformedLayer->outData.size() != 0) ? transformedLayer->outData[0]->getPrecision()
                                                                  : Precision::UNSPECIFIED)
                      << std::endl;
        }

#endif
    }

    // Step #3: cleanup transformations execution
    for (size_t i = 0; i < context.getLayers().size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      for (size_t i = 0; i < context.getLayers().size(); ++i) {" << std::endl;
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (layer == nullptr) {" << std::endl;
            continue;
        }

        const auto it = transformations.cleanupTransformations.find(layer->type);
        if (it != transformations.cleanupTransformations.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:          if (it != transformations.cleanupTransformations.end()) {" << std::endl;
            it->second->transform(context, *layer);
        }
    }
}

std::vector<Precision> LowPrecisionTransformer::getPrecisionsOnActivations(const std::string& layerType) const noexcept {
    const LayerTransformationPtr transformation = transformations.find(layerType);
    if (transformation == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      if (transformation == nullptr) {" << std::endl;
        return std::vector<Precision>();
    }
    return transformation->getPrecisionsOnActivations();
}

bool LowPrecisionTransformer::isQuantized(const CNNLayer& layer) const noexcept {
    const LayerTransformationPtr transformation = transformations.find(layer.type);
    if (transformation == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      if (transformation == nullptr) {" << std::endl;
        return false;
    }
    return transformation->isQuantized(layer);
}

bool LowPrecisionTransformer::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    const LayerTransformationPtr transformation = transformations.find(layer.type);
    if (transformation == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/transformer.cpp:      if (transformation == nullptr) {" << std::endl;
        return false;
    }
    return transformation->isPrecisionPreserved(layer);
}
