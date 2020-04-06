#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/layer_transformation.hpp"
#include "low_precision_transformations/network_helper.hpp"

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
#include <utility>
#include <unordered_set>
#include <vector>

#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

const char LayerTransformation::lastLayerPrefix[] = "_original";

LayerTransformation::LayerTransformation(const Params& params) :
    updatePrecisions(params.updatePrecisions),
    quantizeOutputs(params.quantizeOutputs),
    weightsToConst(params.weightsToConst),
    quantizedTensorAlignmentOnActivations(params.quantizedTensorAlignmentOnActivations),
    quantizedTensorAlignmentOnWeights(params.quantizedTensorAlignmentOnWeights),
    roundQuantizedValues(params.roundQuantizedValues),
    updateBiases(params.updateBiases),
    supportAsymmetricQuantization(params.supportAsymmetricQuantization),
    precisionsOnActivations(params.precisionsOnActivations),
    precisionsOnWeights(params.precisionsOnWeights),
    layerTransformationsManager(nullptr),
    paramsManager(nullptr),
    quantizationIntervalSymmetryRatioThreshold(1.e-5f),
    quantizationIntervalAsymmetryThreshold(1.e-6),
    zeroThreshold(1.e-6f),
    dequantizationShiftToZeroRatioTreshold(1.e-6f),
    minQuantizationLevels(2ul) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      minQuantizationLevels(2ul) {" << std::endl;}

void LayerTransformation::setParamsManager(IParamsManager* paramsManager) noexcept {
    this->paramsManager = paramsManager;
}

void LayerTransformation::setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept {
    this->layerTransformationsManager = layerTransformationsManager;
}

void LayerTransformation::setUpdatePrecisions(const bool updatePrecisions) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  void LayerTransformation::setUpdatePrecisions(const bool updatePrecisions) {" << std::endl;
    this->updatePrecisions = updatePrecisions;
}

void LayerTransformation::setQuantizeOutputs(const bool quantizeOutputs) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  void LayerTransformation::setQuantizeOutputs(const bool quantizeOutputs) {" << std::endl;
    this->quantizeOutputs = quantizeOutputs;
}

void LayerTransformation::setWeightsToConst(const bool weightsToConst) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  void LayerTransformation::setWeightsToConst(const bool weightsToConst) {" << std::endl;
    this->weightsToConst = weightsToConst;
}

void LayerTransformation::setQuantizedTensorAlignmentOnActivations(
    const QuantizedTensorAlignment quantizedTensorAlignmentOnActivations) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      const QuantizedTensorAlignment quantizedTensorAlignmentOnActivations) {" << std::endl;
    this->quantizedTensorAlignmentOnActivations = quantizedTensorAlignmentOnActivations;
}

void LayerTransformation::setQuantizedTensorAlignmentOnWeights(
    const QuantizedTensorAlignment quantizedTensorAlignmentOnWeights) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      const QuantizedTensorAlignment quantizedTensorAlignmentOnWeights) {" << std::endl;
    this->quantizedTensorAlignmentOnWeights = quantizedTensorAlignmentOnWeights;
}

const std::vector<Precision>& LayerTransformation::getPrecisionsOnActivations() const {
    return precisionsOnActivations;
}

const std::vector<Precision>& LayerTransformation::getPrecisionsOnWeights() const {
    return precisionsOnWeights;
}

bool LayerTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const {
    if (!isQuantized(layer)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (!isQuantized(layer)) {" << std::endl;
        return false;
    }

    if (!quantizeOutputs) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (!quantizeOutputs) {" << std::endl;
        OutputsDataMap outputs;
        context.network.getOutputsInfo(outputs);
        if (outputs.find(layer.name) != outputs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (outputs.find(layer.name) != outputs.end()) {" << std::endl;
            return false;
        }
    }

    return true;
}

Precision LayerTransformation::getPrecisionBeforeParentDequantizationScaleShift(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  Precision LayerTransformation::getPrecisionBeforeParentDequantizationScaleShift(const CNNLayer& layer) {" << std::endl;
    const CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(layer, 0);
    if (scaleShift == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (scaleShift == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "dequantization ScaleShift layer is absent";
    }

    if (scaleShift->type != "ScaleShift") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (scaleShift->type != 'ScaleShift') {" << std::endl;
        THROW_IE_EXCEPTION << "not expected dequantization layer type " << scaleShift->type;
    }

    if (scaleShift->insData.size() < 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (scaleShift->insData.size() < 1) {" << std::endl;
        THROW_IE_EXCEPTION << "is not expected ScaleShift '" << scaleShift->name << "' insert data size "
                           << scaleShift->insData.size();
    }

    const DataWeakPtr insDataWeak = scaleShift->insData[0];
    const DataPtr insData = insDataWeak.lock();
    if (insData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (insData == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "input data is absent";
    }

    return insData->getPrecision();
}

void LayerTransformation::fillFromQuantizationDetails(
    const QuantizationDetails& quantizationDetails,
    const DataPrecision& dataPrecision,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) const {
    // TODO: refactor: make optional
    const float minQuantizationScale = 1e-32f;
    const float maxQuantizationScale = 1e32f;

    bool denormalOutputValuesWasUpdated = false;
    dequantizationScales.resize(quantizationDetails.outputChannelsCount);
    dequantizationShifts.resize(quantizationDetails.outputChannelsCount);

    for (size_t channel = 0lu; channel < quantizationDetails.outputChannelsCount; ++channel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      for (size_t channel = 0lu; channel < quantizationDetails.outputChannelsCount; ++channel) {" << std::endl;
        float dequantizationScale = 0.f;
        float dequantizationShift = 0.f;
        if (dataPrecision.precision.isSigned()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (dataPrecision.precision.isSigned()) {" << std::endl;
            // I8
            dequantizationScale =
                (quantizationDetails.getOutputHighValue(channel) - quantizationDetails.getOutputLowValue(channel)) /
                (dataPrecision.max - dataPrecision.min);
            const float quantValue =
                (quantizationDetails.getOutputHighValue(channel) - quantizationDetails.getOutputLowValue(channel)) /
                (dataPrecision.max - dataPrecision.min);

            const float actualLowPartQuantValue =
                std::fabs(quantizationDetails.getOutputLowValue(channel) / dataPrecision.min);
            const float actualHighPartQuantValue =
                std::fabs(quantizationDetails.getOutputHighValue(channel) / dataPrecision.max);

            if (actualLowPartQuantValue < actualHighPartQuantValue) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:              if (actualLowPartQuantValue < actualHighPartQuantValue) {" << std::endl;
                dequantizationShift =
                    quantizationDetails.getOutputLowValue(channel) - dataPrecision.min * quantValue;
            } else {
                dequantizationShift =
                    quantizationDetails.getOutputHighValue(channel) - dataPrecision.max * quantValue;
            }
        } else {
            // U8
            dequantizationScale =
                (quantizationDetails.getOutputHighValue(channel) - quantizationDetails.getOutputLowValue(channel)) /
                (dataPrecision.max - dataPrecision.min);
            dequantizationShift = quantizationDetails.getOutputLowValue(channel);
        }

        if (fabs(dequantizationScale) < minQuantizationScale) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (fabs(dequantizationScale) < minQuantizationScale) {" << std::endl;
            dequantizationScales[channel] = minQuantizationScale;
            denormalOutputValuesWasUpdated = true;
        } else if (fabs(dequantizationScale) > maxQuantizationScale) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          } else if (fabs(dequantizationScale) > maxQuantizationScale) {" << std::endl;
            dequantizationScales[channel] = dequantizationScale > 0.f ? maxQuantizationScale : -maxQuantizationScale;
            denormalOutputValuesWasUpdated = true;
        } else {
            dequantizationScales[channel] = dequantizationScale;
        }

        dequantizationShifts[channel] = dequantizationShift;
    }

    checkAndUpdateDequantizationShiftWithZero(quantizationDetails, dequantizationShifts);
}

void LayerTransformation::checkAndUpdateDequantizationShiftWithZero(
    const QuantizationDetails& quantizationDetails,
    std::vector<float>& dequantizationShifts) const {
    auto compare = [](float value1, float value2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      auto compare = [](float value1, float value2) {" << std::endl; return (std::fabs(value1) < std::fabs(value2)); };

    const auto maxShiftIt = std::max_element(dequantizationShifts.begin(), dequantizationShifts.end(), compare);
    if (maxShiftIt == dequantizationShifts.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (maxShiftIt == dequantizationShifts.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected dequantization shifts max value";
    }

    const auto maxOutputLowIt = std::max_element(quantizationDetails.outputLowValues.begin(), quantizationDetails.outputLowValues.end(), compare);
    if (maxOutputLowIt == quantizationDetails.outputLowValues.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (maxOutputLowIt == quantizationDetails.outputLowValues.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected dequantization output low value";
    }

    const auto maxOutputHighIt = std::max_element(quantizationDetails.outputHighValues.begin(), quantizationDetails.outputHighValues.end(), compare);
    if (maxOutputHighIt == quantizationDetails.outputHighValues.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (maxOutputHighIt == quantizationDetails.outputHighValues.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected dequantization output high value";
    }

    const float maxOutputIt = std::max(std::fabs(*maxOutputLowIt), std::fabs(*maxOutputHighIt));
    const float relative = std::fabs(*maxShiftIt) / std::fabs(maxOutputIt);
    if (relative < dequantizationShiftToZeroRatioTreshold) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (relative < dequantizationShiftToZeroRatioTreshold) {" << std::endl;
        std::fill(dequantizationShifts.begin(), dequantizationShifts.end(), 0.f);
    }
}

void LayerTransformation::fillFromDequantizationLayer(
    const CNNLayer& dequantizationLayer,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) const {
    if (dequantizationLayer.type != "ScaleShift") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (dequantizationLayer.type != 'ScaleShift') {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected dequantization layer type " << dequantizationLayer.type;
    }

    CNNLayerPtr dequantizationLayerPtr = std::make_shared<CNNLayer>(dequantizationLayer);
    Blob::Ptr weightsBlob = CNNNetworkHelper::getBlob(dequantizationLayerPtr, "weights");
    const auto weightsBuffer = CNNNetworkHelper::getFloatData(weightsBlob);

    Blob::Ptr shiftsBlob = CNNNetworkHelper::getBlob(dequantizationLayerPtr, "biases");
    const auto shiftsBuffer = CNNNetworkHelper::getFloatData(shiftsBlob);

    const size_t inputCannelsCount = CNNNetworkHelper::getInputChannelsCount(dequantizationLayer);
    dequantizationScales.resize(inputCannelsCount);
    dequantizationShifts.resize(inputCannelsCount);
    for (size_t channel = 0; channel < inputCannelsCount; ++channel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      for (size_t channel = 0; channel < inputCannelsCount; ++channel) {" << std::endl;
        dequantizationScales[channel] = (weightsBlob->size() == 1ul) ? weightsBuffer.get()[0] : weightsBuffer.get()[channel];
        dequantizationShifts[channel] = (shiftsBlob->size() == 1ul) ? shiftsBuffer.get()[0] : shiftsBuffer.get()[channel];
    }
}

void LayerTransformation::setQuantizationIntervalSymmetryRatioThreshold(const float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  void LayerTransformation::setQuantizationIntervalSymmetryRatioThreshold(const float value) {" << std::endl;
    this->quantizationIntervalSymmetryRatioThreshold = value;
}

void LayerTransformation::setQuantizationIntervalAsymmetryThreshold(const float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  void LayerTransformation::setQuantizationIntervalAsymmetryThreshold(const float value) {" << std::endl;
    this->quantizationIntervalAsymmetryThreshold = value;
}

void LayerTransformation::setZeroThreshold(const float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  void LayerTransformation::setZeroThreshold(const float value) {" << std::endl;
    this->zeroThreshold = value;
}

void LayerTransformation::setDequantizationShiftToZeroRatioTreshold(const float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  void LayerTransformation::setDequantizationShiftToZeroRatioTreshold(const float value) {" << std::endl;
    this->dequantizationShiftToZeroRatioTreshold = value;
}

void LayerTransformation::setMinQuantizationLevels(const size_t levels) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  void LayerTransformation::setMinQuantizationLevels(const size_t levels) {" << std::endl;
    this->minQuantizationLevels = levels;
}

Precision LayerTransformation::getPrecisionParent(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:  Precision LayerTransformation::getPrecisionParent(const CNNLayer& layer) {" << std::endl;
    const CNNLayerPtr parent = CNNNetworkHelper::getParent(layer, 0);
    if (parent == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (parent == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "parent layer is absent";
    }

    for (const DataPtr outData : parent->outData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      for (const DataPtr outData : parent->outData) {" << std::endl;
        const auto inputTo = outData->getInputTo();
        for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {" << std::endl;
            if (it->second->name == layer.name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:              if (it->second->name == layer.name) {" << std::endl;
                return outData->getPrecision();
            }
        }
    }

    THROW_IE_EXCEPTION << "out data from '" << parent->name << "' to '" << layer.name << "' was not found";
}

LayerTransformation::PrecisionDetails LayerTransformation::getPrecisionDetails(const QuantizationDetails& quantizationDetails) const {
    const float asymmetricIntervalSideRatio256 = -128.f / 127.f;
    bool hasNegative = false;
    bool signedPrecision = true;
    bool unsignedPrecision = true;

    bool hasZeroPoint = false;
    for (size_t i = 0; i < quantizationDetails.outputLowValues.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      for (size_t i = 0; i < quantizationDetails.outputLowValues.size(); ++i) {" << std::endl;
        const bool signedInterval = std::signbit(quantizationDetails.outputLowValues[i]) != std::signbit(quantizationDetails.outputHighValues[i]);
        const bool boundaryValuesAreNotZero =
            (std::fabs(quantizationDetails.outputLowValues[i]) >= zeroThreshold) &&
            (std::fabs(quantizationDetails.outputHighValues[i]) >= zeroThreshold);
        if (signedInterval && boundaryValuesAreNotZero) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (signedInterval && boundaryValuesAreNotZero) {" << std::endl;
            // signed
            unsignedPrecision = false;
            hasNegative = true;

            if (quantizationDetails.levels == 256) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:              if (quantizationDetails.levels == 256) {" << std::endl;
                const float ratio = quantizationDetails.outputLowValues[i] / quantizationDetails.outputHighValues[i];
                if (std::fabs(ratio - asymmetricIntervalSideRatio256) > quantizationIntervalAsymmetryThreshold) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:                  if (std::fabs(ratio - asymmetricIntervalSideRatio256) > quantizationIntervalAsymmetryThreshold) {" << std::endl;
                    hasZeroPoint = true;
                }
            }

            if (quantizationDetails.levels == 255) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:              if (quantizationDetails.levels == 255) {" << std::endl;
                const float threshold = std::min(
                    std::fabs(quantizationDetails.outputLowValues[i]),
                    std::fabs(quantizationDetails.outputHighValues[i])) * quantizationIntervalSymmetryRatioThreshold;
                if (std::fabs(std::fabs(quantizationDetails.outputHighValues[i]) - std::fabs(quantizationDetails.outputLowValues[i])) > threshold) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:                  if (std::fabs(std::fabs(quantizationDetails.outputHighValues[i]) - std::fabs(quantizationDetails.outputLowValues[i])) > threshold) {" << std::endl;
                    hasZeroPoint = true;
                }
            }
        } else {
            // unsigned
            signedPrecision = false;
            if (boundaryValuesAreNotZero) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:              if (boundaryValuesAreNotZero) {" << std::endl;
                hasZeroPoint = boundaryValuesAreNotZero;
            }
        }
    }

    if (!hasZeroPoint) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (!hasZeroPoint) {" << std::endl;
        if (signedPrecision && (!unsignedPrecision)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (signedPrecision && (!unsignedPrecision)) {" << std::endl;
            return LayerTransformation::PrecisionDetails(Precision::I8, hasNegative, hasZeroPoint);
        }

        if ((!signedPrecision) && unsignedPrecision) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if ((!signedPrecision) && unsignedPrecision) {" << std::endl;
            return LayerTransformation::PrecisionDetails(Precision::U8, hasNegative, hasZeroPoint);
        }
    }

    return LayerTransformation::PrecisionDetails(Precision::UNSPECIFIED, hasNegative, hasZeroPoint);
}

bool LayerTransformation::isQuantized(const CNNLayer& layer) const noexcept {
    return true;
}

bool LayerTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return true;
}

DataPrecision LayerTransformation::getDataPrecision(
    const CNNLayer& layer,
    const QuantizationDetails& quantizationDetails,
    const bool onWeights,
    const bool supportAsymmetricQuantization) const {
    std::vector<Precision> precisions;
    if (onWeights) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (onWeights) {" << std::endl;
        precisions = precisionsOnWeights;
    } else {
        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
        if (children.size() != 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (children.size() != 0) {" << std::endl;
            precisions = paramsManager->getPrecisionsOnActivations(children[0]->type);
        } else {
            precisions = precisionsOnActivations;
        }
    }

    {
        PrecisionDetails precisionDetailsAtOutputIntervals = getPrecisionDetails(quantizationDetails);
        if (precisionDetailsAtOutputIntervals.precision != Precision::UNSPECIFIED) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (precisionDetailsAtOutputIntervals.precision != Precision::UNSPECIFIED) {" << std::endl;
            if (!onWeights) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:              if (!onWeights) {" << std::endl;
                fillAvailablePrecisions(layer, precisions);
            }

            // if supportedPrecisions is empty then use the first available, not supported layer will be in original precision
            if (!precisions.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:              if (!precisions.empty()) {" << std::endl;
                const auto foundIt = std::find(precisions.begin(), precisions.end(), precisionDetailsAtOutputIntervals.precision);
                const Precision resultPrecision = foundIt != precisions.end() ?
                    precisionDetailsAtOutputIntervals.precision :
                    *precisions.begin();
                return DataPrecision(
                    resultPrecision,
                    DataPrecision::getMinValue(resultPrecision, quantizationDetails.levels),
                    DataPrecision::getMaxValue(resultPrecision));
            }
        }
    }

    if (precisions.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (precisions.empty()) {" << std::endl;
        return DataPrecision(Precision::UNSPECIFIED, 0.f, 0.f);
    }

    const Precision precision = *precisions.begin();
    return DataPrecision(
        precision,
        DataPrecision::getMinValue(precision, quantizationDetails.levels),
        DataPrecision::getMaxValue(precision));
}

void LayerTransformation::fillAvailablePrecisions(const CNNLayer& layer, std::vector<Precision>& availablePrecisions) const {
    if (availablePrecisions.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      if (availablePrecisions.empty()) {" << std::endl;
        return;
    }

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    for (CNNLayerPtr child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:      for (CNNLayerPtr child : children) {" << std::endl;
        if (!layerTransformationsManager->isQuantized(*child)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (!layerTransformationsManager->isQuantized(*child)) {" << std::endl;
            // low precision chain is interrupted here: next layer supported precisions are ignored
            continue;
        }

        const std::vector<Precision> childPrecisionsOnActivations = paramsManager->getPrecisionsOnActivations(child->type);
        if (childPrecisionsOnActivations.size() == 0ul) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (childPrecisionsOnActivations.size() == 0ul) {" << std::endl;
            continue;
        }

        for (size_t index = 0ul; index < availablePrecisions.size();) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          for (size_t index = 0ul; index < availablePrecisions.size();) {" << std::endl;
            const Precision availablePrecision = availablePrecisions[index];
            if (!std::any_of(
                childPrecisionsOnActivations.begin(),
                childPrecisionsOnActivations.end(),
                [&](const Precision precision) { return availablePrecision == precision; })) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:                  [&](const Precision precision) { return availablePrecision == precision; })) {" << std::endl;
                availablePrecisions.erase(availablePrecisions.begin() + index);
            } else {
                ++index;
            }
        }

        if (!layerTransformationsManager->isPrecisionPreserved(*child)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (!layerTransformationsManager->isPrecisionPreserved(*child)) {" << std::endl;
            continue;
        }

        fillAvailablePrecisions(*child, availablePrecisions);
        if (availablePrecisions.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/layer_transformation.cpp:          if (availablePrecisions.empty()) {" << std::endl;
            return;
        }
    }
}
