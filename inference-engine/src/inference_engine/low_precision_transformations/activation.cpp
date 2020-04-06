#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/activation.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void ActivationTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!CaselessEq<std::string>()(layer.type, "ReLU")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:      if (!CaselessEq<std::string>()(layer.type, 'ReLU')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer type '" << layer.name << "' is not correct";
    }

    const CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(layer, 0);
    if ((scaleShift == nullptr) || (scaleShift->type != "ScaleShift")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:      if ((scaleShift == nullptr) || (scaleShift->type != 'ScaleShift')) {" << std::endl;
        return;
    }

    // TODO: temporary limitation
    if (scaleShift->insData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:      if (scaleShift->insData.size() != 1) {" << std::endl;
        return;
    }

    const Blob::Ptr weightsBlob = CNNNetworkHelper::getBlob(scaleShift, "weights");
    auto weights = CNNNetworkHelper::getFloatData(weightsBlob);
    const std::vector<float> scales = std::vector<float>(weights.get(), weights.get() + weightsBlob->size());

    const Blob::Ptr biasesBlob = CNNNetworkHelper::getBlob(scaleShift, "biases");
    auto biases = CNNNetworkHelper::getFloatData(biasesBlob);
    const std::vector<float> shifts = std::vector<float>(biases.get(), biases.get() + biasesBlob->size());

    CNNLayerPtr activationLayer;
    if ((std::all_of(shifts.begin(), shifts.end(),
                     [](float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:                       [](float value) {" << std::endl;
                         return value == 0.0;
                     })) &&
        (std::all_of(scales.begin(), scales.end(), [](float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:          (std::all_of(scales.begin(), scales.end(), [](float value) {" << std::endl;
            return value >= 0.0;
        }))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:          }))) {" << std::endl;
        activationLayer = std::make_shared<CNNLayer>(layer);
    } else {
        const float negativeSlope = layer.GetParamAsFloat("negative_slope", 0.0);
        if (negativeSlope != 0.0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:          if (negativeSlope != 0.0) {" << std::endl;
            return;
        }

        if (!(std::equal(shifts.begin() + 1, shifts.end(), shifts.begin())) ||
            !(std::equal(scales.begin() + 1, scales.end(), scales.begin()))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:              !(std::equal(scales.begin() + 1, scales.end(), scales.begin()))) {" << std::endl;
            return;
        }

        const Precision precision = getPrecisionBeforeParentDequantizationScaleShift(layer);

        std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*scaleShift);
        if (parents.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:          if (parents.size() != 1) {" << std::endl;
            return;
        }

        LayerParams layerParams {layer.name + "_Clamp", "Clamp", precision};
        activationLayer = std::make_shared<ClampLayer>(layerParams);

        ClampLayer* clampLayer = dynamic_cast<ClampLayer*>(activationLayer.get());
        if (std::all_of(scales.begin(), scales.end(), [](float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:          if (std::all_of(scales.begin(), scales.end(), [](float value) {" << std::endl;
                return value >= 0.0;
            })) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:              })) {" << std::endl;
            clampLayer->min_value = -shifts[0] / scales[0];
            clampLayer->max_value = DataPrecision::getMaxValue(precision);
            clampLayer->params["min"] = std::to_string(clampLayer->min_value);
            clampLayer->params["max"] = std::to_string(clampLayer->max_value);
        } else {
            // TODO: workaround: only U8 on activations
            clampLayer->min_value = DataPrecision::getMinValue(precision, 256);
            clampLayer->max_value = -shifts[0] / scales[0];
            clampLayer->params["min"] = std::to_string(clampLayer->min_value);
            clampLayer->params["max"] = std::to_string(clampLayer->max_value);
        }

        std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
        if (children.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:          if (children.size() != 1) {" << std::endl;
            return;
        }

        for (CNNLayerPtr child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:          for (CNNLayerPtr child : children) {" << std::endl;
            CNNNetworkHelper::addLayer(context, std::make_shared<CNNLayer>(layer), child, activationLayer);
        }

        CNNNetworkHelper::removeLayer(context.network, std::make_shared<CNNLayer>(layer));
        context.removeLayer(layer);
    }

    if (updatePrecisions) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:      if (updatePrecisions) {" << std::endl;
        CNNNetworkHelper::setOutDataPrecision(layer, getPrecisionBeforeParentDequantizationScaleShift(layer));
    }

    CNNNetworkHelper::removeLayer(context.network, scaleShift);
    context.removeLayer(*scaleShift);

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*activationLayer);
    for (const CNNLayerPtr& child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/activation.cpp:      for (const CNNLayerPtr& child : children) {" << std::endl;
        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(context, activationLayer, child,
                                                                                 DequantizationDetails(scales, shifts));
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    }
}
