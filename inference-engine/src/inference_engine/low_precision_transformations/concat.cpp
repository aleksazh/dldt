#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat.hpp"
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/quantization_details.hpp"

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
#include <vector>

#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool ConcatTransformation::getQuantizeLayers(
    CNNLayerPtr layer,
    std::vector<std::string>& childNameOurAfterQuantizeLayers,
    std::vector<CNNLayerPtr>& quantizeLayers,
    std::vector<std::vector<CNNLayerPtr>>& intermediateLayers,
    std::vector<CNNLayerPtr>& concatLayers,
    std::string childName,
    std::vector<CNNLayerPtr>& sideOutputLayers,
    std::vector<std::string>& childrenNameSideOutputLayers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      std::vector<std::string>& childrenNameSideOutputLayers) {" << std::endl;
    if (!CaselessEq<std::string>()(layer->type, "FakeQuantize") &&
        !CaselessEq<std::string>()(layer->type, "Quantize")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          !CaselessEq<std::string>()(layer->type, 'Quantize')) {" << std::endl;
        do {
            if (CaselessEq<std::string>()(layer->type, "Pooling")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:              if (CaselessEq<std::string>()(layer->type, 'Pooling')) {" << std::endl;
                intermediateLayers.back().push_back(layer);
                childName = layer->name;
                layer = CNNNetworkHelper::getParent(*layer, 0);
            } else if (CaselessEq<std::string>()(layer->type, "Concat")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:              } else if (CaselessEq<std::string>()(layer->type, 'Concat')) {" << std::endl;
                concatLayers.push_back(layer);

                if (layer->outData[0]->getInputTo().size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:                  if (layer->outData[0]->getInputTo().size() != 1) {" << std::endl;
                    sideOutputLayers.push_back(layer);
                    childrenNameSideOutputLayers.push_back(childName);
                }
                int size = layer->insData.size();
                childName = layer->name;
                for (int i = 0; i < size; i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:                  for (int i = 0; i < size; i++) {" << std::endl;
                    CNNLayerPtr layer1 = CNNNetworkHelper::getParent(*layer, i);
                    intermediateLayers.push_back({});
                    if (!getQuantizeLayers(
                        layer1,
                        childNameOurAfterQuantizeLayers,
                        quantizeLayers,
                        intermediateLayers,
                        concatLayers,
                        childName,
                        sideOutputLayers,
                        childrenNameSideOutputLayers)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:                          childrenNameSideOutputLayers)) {" << std::endl;
                        return false;
                    }
                }
                return true;
            } else {
                return false;
            }
        } while (!CaselessEq<std::string>()(layer->type, "FakeQuantize") &&
                 !CaselessEq<std::string>()(layer->type, "Quantize"));
    }

    childNameOurAfterQuantizeLayers.push_back(childName);
    quantizeLayers.push_back(layer);
    return true;
}

void ConcatTransformation::transform(TransformationContext& context, CNNLayer& concat) const {
    if (!canBeTransformed(context, concat)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if (!canBeTransformed(context, concat)) {" << std::endl;
        return;
    }

    if (!CaselessEq<std::string>()(concat.type, "Concat")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if (!CaselessEq<std::string>()(concat.type, 'Concat')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer type '" << concat.name << "' is not correct";
    }

    if (concat.GetParamAsUInt("axis", 1) != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if (concat.GetParamAsUInt('axis', 1) != 1) {" << std::endl;
        return;
    }

    if ((concat.insData.size() < 2)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if ((concat.insData.size() < 2)) {" << std::endl;
        THROW_IE_EXCEPTION << "layer inputs '" << concat.insData.size() << "' is not correct";
    }

    std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(concat);
    if (CNNNetworkHelper::IsChild(children, {"Concat"}, {"Pooling"})) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if (CNNNetworkHelper::IsChild(children, {'Concat'}, {'Pooling'})) {" << std::endl;
        return;
    }

    std::vector<CNNLayerPtr> quantizeLayers;
    std::vector<std::vector<CNNLayerPtr>> intermediateLayers;
    std::vector<CNNLayerPtr> concatLayers;
    const auto inputDataNumber = concat.insData.size();
    std::vector<std::string> childNameOurAfterQuantizeLayers;
    std::vector<QuantizationDetails> quantizationLayersDetails;
    std::vector<CNNLayerPtr> sideOutputLayers;
    std::vector<std::string> childrenNameSideOutputLayers;
    for (size_t index = 0lu; index < inputDataNumber; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      for (size_t index = 0lu; index < inputDataNumber; index++) {" << std::endl;
        DataPtr quantizeOnData = concat.insData[index].lock();
        if (quantizeOnData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          if (quantizeOnData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input is absent";
        }
        auto parentLayer = quantizeOnData->getCreatorLayer().lock();
        intermediateLayers.push_back({});
        if (!getQuantizeLayers(
            parentLayer,
            childNameOurAfterQuantizeLayers,
            quantizeLayers,
            intermediateLayers,
            concatLayers,
            concat.name,
            sideOutputLayers,
            childrenNameSideOutputLayers)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:              childrenNameSideOutputLayers)) {" << std::endl;
            return;
        }
    }

    if (quantizeLayers.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if (quantizeLayers.empty()) {" << std::endl;
        return;
    }

    size_t quantizationLevels = 0lu;
    for (int i = 0; i < quantizeLayers.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      for (int i = 0; i < quantizeLayers.size(); i++) {" << std::endl;
        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(*quantizeLayers[i]);
        if (!QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) continue;
        if (quantizationLevels == 0lu) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          if (quantizationLevels == 0lu) {" << std::endl;
            quantizationLevels = quantizationDetails.levels;
        } else if (quantizationLevels != quantizationDetails.levels) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          } else if (quantizationLevels != quantizationDetails.levels) {" << std::endl;
            THROW_IE_EXCEPTION << "different quantization levels " << quantizationLevels << " are not supported";
        }

        quantizationLayersDetails.push_back(quantizationDetails);
    }

    const DataPrecision dataPrecision = getDataPrecision(*quantizeLayers[0], QuantizationDetails::getDetails(*quantizeLayers[0]), false, false);
    if (dataPrecision.precision == Precision::UNSPECIFIED) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if (dataPrecision.precision == Precision::UNSPECIFIED) {" << std::endl;
        return;
    }

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(concat);

    dequantizationScales.resize(outputChannelsCount);
    dequantizationShifts.resize(outputChannelsCount);
    const auto parentsCount = quantizeLayers.size();
    std::vector<std::vector<float>> dequantizationShiftsLayers;
    dequantizationShiftsLayers.resize(parentsCount);

    if ((quantizationLayersDetails[0].inputHighValues.size() == 1)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if ((quantizationLayersDetails[0].inputHighValues.size() == 1)) {" << std::endl;
        float outputLowValue = quantizationLayersDetails[0].outputLowValues[0];
        float outputHighValue = quantizationLayersDetails[0].outputHighValues[0];
        for (size_t index = 0lu; index < parentsCount; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          for (size_t index = 0lu; index < parentsCount; index++) {" << std::endl;
            if (outputLowValue > quantizationLayersDetails[index].outputLowValues[0]) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:              if (outputLowValue > quantizationLayersDetails[index].outputLowValues[0]) {" << std::endl;
                outputLowValue = quantizationLayersDetails[index].outputLowValues[0];
            }
            if (outputHighValue < quantizationLayersDetails[index].outputHighValues[0]) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:              if (outputHighValue < quantizationLayersDetails[index].outputHighValues[0]) {" << std::endl;
                outputHighValue = quantizationLayersDetails[index].outputHighValues[0];
            }
        }

        const float maxOutputInterval = outputHighValue - outputLowValue;
        if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {" << std::endl;
            const size_t minLevels = getMinQuantizationLevels(dataPrecision, maxOutputInterval, quantizationLayersDetails, outputLowValue);
            if (minLevels < this->minQuantizationLevels) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:              if (minLevels < this->minQuantizationLevels) {" << std::endl;
                return;
            }
        }

        const float generalScaleDequantize = maxOutputInterval / (dataPrecision.max - dataPrecision.min);

        for (int index = 0; index < parentsCount; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          for (int index = 0; index < parentsCount; index++) {" << std::endl;
            if (quantizeLayers[index] == nullptr)
                continue;
            CNNLayer& fakeQuantizeLayer = *quantizeLayers[index];
            const QuantizationDetails quantizationDetails = quantizationLayersDetails[index];

            // TODO: copy/paste, refactor: extract to MultiBranchTransformation::updateQuantizationRange
            switch (quantizedTensorAlignmentOnActivations) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:              switch (quantizedTensorAlignmentOnActivations) {" << std::endl;
            case QuantizedTensorAlignment::None: {
                const float quantizationScale = (dataPrecision.max - dataPrecision.min) / maxOutputInterval;

                const float inputLowValue =
                    (quantizationDetails.outputLowValues[0] - outputLowValue) * quantizationScale;
                const float inputHighValue =
                    (quantizationDetails.outputHighValues[0] - outputLowValue) * quantizationScale;

                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3,
                                              updatePrecisions ? roundf(inputLowValue) : inputLowValue);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4,
                                              updatePrecisions ? roundf(inputHighValue) : inputHighValue);
                break;
            }
            case QuantizedTensorAlignment::UpdateIntervals: {
                const float inputLowValue = quantizationDetails.outputLowValues[0] != 0.0
                                                ? (quantizationDetails.inputLowValues[0] *
                                                   (outputLowValue / quantizationDetails.outputLowValues[0]))
                                                : outputLowValue;
                const float inputHighValue = quantizationDetails.outputHighValues[0] != 0.0
                                                 ? (quantizationDetails.inputHighValues[0] *
                                                    (outputHighValue / quantizationDetails.outputHighValues[0]))
                                                 : outputHighValue;

                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 1, inputLowValue);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 2, inputHighValue);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, dataPrecision.min);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, dataPrecision.max);

                break;
            }
            case QuantizedTensorAlignment::UpdateLevel: {
                const float quantizationScale = (dataPrecision.max - dataPrecision.min) / maxOutputInterval;

                const float inputLowValue = roundf((quantizationDetails.outputLowValues[0] - outputLowValue) * quantizationScale);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, updatePrecisions ? roundf(inputLowValue) : inputLowValue);

                const float inputHighValue = roundf((quantizationDetails.outputHighValues[0] - outputLowValue) * quantizationScale);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, updatePrecisions ? roundf(inputHighValue) : inputHighValue);

                const int levels = static_cast<int>(fabs(inputHighValue - inputLowValue) + 1.0);
                fakeQuantizeLayer.params["levels"] = std::to_string(levels);
                QuantizeLayer* layer = dynamic_cast<QuantizeLayer*>(&fakeQuantizeLayer);
                if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:                  if (layer == nullptr) {" << std::endl;
                    THROW_IE_EXCEPTION << "incorrect type for layer " << fakeQuantizeLayer.name;
                }
                layer->levels = levels;

                break;
            }
            default: {
                THROW_IE_EXCEPTION << "unexpected value " << quantizedTensorAlignmentOnActivations;
            }
            }

            if (updatePrecisions) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:              if (updatePrecisions) {" << std::endl;
                CNNNetworkHelper::setOutDataPrecision(fakeQuantizeLayer, dataPrecision.precision);

                const std::vector<CNNLayerPtr>& intermediateLayersList = intermediateLayers[index];
                for (const CNNLayerPtr intermediateLayer : intermediateLayersList) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:                  for (const CNNLayerPtr intermediateLayer : intermediateLayersList) {" << std::endl;
                    CNNNetworkHelper::setOutDataPrecision(*intermediateLayer, dataPrecision.precision);
                }
            }

            dequantizationShiftsLayers[index].push_back(outputLowValue);
        }

        for (size_t channel = 0lu; channel < outputChannelsCount; ++channel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          for (size_t channel = 0lu; channel < outputChannelsCount; ++channel) {" << std::endl;
            dequantizationScales[channel] = generalScaleDequantize;
            dequantizationShifts[channel] = outputLowValue;
        }
    } else {
        return;
    }

    // Add scaleshift at other outputs of the Quantize layer
    for (int index = 0; index < parentsCount; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      for (int index = 0; index < parentsCount; index++) {" << std::endl;
        context.quantizedFakeQuantizeNames.insert(quantizeLayers[index]->name);
        if (quantizeLayers[index]->outData[0]->getInputTo().size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          if (quantizeLayers[index]->outData[0]->getInputTo().size() != 1) {" << std::endl;
            std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*quantizeLayers[index], childNameOurAfterQuantizeLayers[index]);

            for (int i = 0; i < children.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:              for (int i = 0; i < children.size(); i++) {" << std::endl;
                const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*quantizeLayers[index]);
                std::vector<float> branchDequantizationScales(outputChannelsCount, dequantizationScales[0]);
                std::vector<float> branchDequantizationShifts(outputChannelsCount, dequantizationShiftsLayers[index][0]);
                CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                    context,
                    std::make_shared<CNNLayer>(*quantizeLayers[index]),
                    children[i],
                    DequantizationDetails(branchDequantizationScales, branchDequantizationShifts, outputChannelsCount));
                context.dequantizationLayersNames.insert(dequantizationLayer->name);
            }
        }
    }

    if (updatePrecisions) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if (updatePrecisions) {" << std::endl;
        CNNNetworkHelper::setOutDataPrecision(concat, dataPrecision.precision);
        for (const CNNLayerPtr concatLayer : concatLayers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          for (const CNNLayerPtr concatLayer : concatLayers) {" << std::endl;
            // TODO: check if the same precision is used: U8 or S8 for all concat layers
            CNNNetworkHelper::setOutDataPrecision(*concatLayer, dataPrecision.precision);
        }
    }

    // Add scaleshift at outputs of our layers
    children = CNNNetworkHelper::getChildren(concat);
    if (children.size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      if (children.size() == 0) {" << std::endl;
        const std::string originalName = concat.name;
        CNNNetworkHelper::renameLayer(context.network, concat.name, concat.name + LayerTransformation::lastLayerPrefix);

        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context,
            std::make_shared<CNNLayer>(concat),
            nullptr,
            DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount), originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          for (const CNNLayerPtr child : children) {" << std::endl;
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(concat),
                child,
                DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }

    // Add scaleshift at outputs of side branches
    for (int index = 0; index < sideOutputLayers.size(); index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      for (int index = 0; index < sideOutputLayers.size(); index++) {" << std::endl;
        const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*sideOutputLayers[index]);
        std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*sideOutputLayers[index], childrenNameSideOutputLayers[index]);
        for (int i = 0; i < children.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          for (int i = 0; i < children.size(); i++) {" << std::endl;
            std::vector<float> dequantizationScales1(outputChannelsCount, dequantizationScales[0]);
            std::vector<float> dequantizationShifts1(outputChannelsCount, dequantizationShifts[0]);
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(*sideOutputLayers[index]),
                children[i],
                DequantizationDetails(dequantizationScales1, dequantizationShifts1, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}

size_t ConcatTransformation::getMinQuantizationLevels(
    const DataPrecision& dataPrecision,
    const float maxOutputInterval,
    const std::vector<QuantizationDetails>& quantizationLayersDetails,
    const float outputLowValue) const {
    size_t minLevels = std::numeric_limits<std::size_t>::max();
    for (const QuantizationDetails quantizationDetails : quantizationLayersDetails) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:      for (const QuantizationDetails quantizationDetails : quantizationLayersDetails) {" << std::endl;
        const float quantizationScale = (dataPrecision.max - dataPrecision.min) / maxOutputInterval;
        const float inputLowValue = roundf((quantizationDetails.outputLowValues[0] - outputLowValue) * quantizationScale);
        const float inputHighValue = roundf((quantizationDetails.outputHighValues[0] - outputLowValue) * quantizationScale);

        const int levels = static_cast<int>(fabs(inputHighValue - inputLowValue) + 1.0);
        if (minLevels > levels) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat.cpp:          if (minLevels > levels) {" << std::endl;
            minLevels = levels;
        }
    }
    return minLevels;
}
