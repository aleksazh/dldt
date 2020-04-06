#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_multi_channels.hpp"
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/quantization_details.hpp"

#include <data_stats.h>
#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"
#include "network_serializer.h"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

size_t getQuantizationLevel(const std::vector<CNNLayerPtr>& fakeQuantizeLayers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:  size_t getQuantizationLevel(const std::vector<CNNLayerPtr>& fakeQuantizeLayers) {" << std::endl;
    size_t quantizationLevels = 0lu;
    for (int i = 0; i < fakeQuantizeLayers.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      for (int i = 0; i < fakeQuantizeLayers.size(); i++) {" << std::endl;
        const CNNLayerPtr fakeQuantizeLayer = fakeQuantizeLayers[i];
        if (fakeQuantizeLayer->type != "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          if (fakeQuantizeLayer->type != 'FakeQuantize') {" << std::endl;
            THROW_IE_EXCEPTION << "not expected layer type " << fakeQuantizeLayer->type;
        }

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(*fakeQuantizeLayer);
        if (!QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          if (!QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) {" << std::endl;
            continue;
        }
        if (quantizationLevels == 0lu) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          if (quantizationLevels == 0lu) {" << std::endl;
            quantizationLevels = quantizationDetails.levels;
        } else if (quantizationLevels != quantizationDetails.levels) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          } else if (quantizationLevels != quantizationDetails.levels) {" << std::endl;
            THROW_IE_EXCEPTION << "different quantization levels " << quantizationLevels << " are not supported";
        }
    }

    return quantizationLevels;
}

bool isCascade(const std::vector<CNNLayerPtr>& concatLayers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:  bool isCascade(const std::vector<CNNLayerPtr>& concatLayers) {" << std::endl;
    for (size_t index = 0ul; index < (concatLayers.size() - 1); ++index) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      for (size_t index = 0ul; index < (concatLayers.size() - 1); ++index) {" << std::endl;
        const CNNLayerPtr childConcatLayer = concatLayers[index];
        const CNNLayerPtr parentConcatLayer = concatLayers[index + 1];
        std::vector<CNNLayerPtr> parents =
            CNNNetworkHelper::getParentsRecursivelyExceptTypes(*childConcatLayer, {"Pooling"});

        bool parentConcatLayerWasFound = false;
        for (const CNNLayerPtr parent : parents) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          for (const CNNLayerPtr parent : parents) {" << std::endl;
            if (parent->name == parentConcatLayer->name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              if (parent->name == parentConcatLayer->name) {" << std::endl;
                parentConcatLayerWasFound = true;
                break;
            }
        }

        if (!parentConcatLayerWasFound) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          if (!parentConcatLayerWasFound) {" << std::endl;
            return false;
        }
    }
    return true;
}

bool isMultiChannel(const std::vector<CNNLayerPtr>& concatLayers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:  bool isMultiChannel(const std::vector<CNNLayerPtr>& concatLayers) {" << std::endl;
    for (const CNNLayerPtr concat : concatLayers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      for (const CNNLayerPtr concat : concatLayers) {" << std::endl;
        const std::vector<CNNLayerPtr> children =
            CNNNetworkHelper::getChildrenRecursivelyExceptTypes(*concat, {"Pooling"});
        if (CNNNetworkHelper::IsChild(children, {"Convolution"})) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          if (CNNNetworkHelper::IsChild(children, {'Convolution'})) {" << std::endl;
            return false;
        }
    }
    return true;
}

bool ConcatMultiChannelsTransformation::getQuantizeLayers(
    CNNLayerPtr layer,
    std::vector<std::string>& childNameOurAfterQuantizeLayers,
    std::vector<CNNLayerPtr>& quantizeLayers,
    std::vector<std::vector<CNNLayerPtr>>& intermediateLayers,
    std::vector<CNNLayerPtr>& concatLayers,
    std::string childName,
    std::vector<CNNLayerPtr>& sideOutputLayers,
    std::vector<std::string>& childrenNameSideOutputLayers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      std::vector<std::string>& childrenNameSideOutputLayers) {" << std::endl;
    if (!CaselessEq<std::string>()(layer->type, "FakeQuantize") &&
        !CaselessEq<std::string>()(layer->type, "Quantize")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          !CaselessEq<std::string>()(layer->type, 'Quantize')) {" << std::endl;
        do {
            if (CaselessEq<std::string>()(layer->type, "Pooling")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              if (CaselessEq<std::string>()(layer->type, 'Pooling')) {" << std::endl;
                intermediateLayers.back().push_back(layer);
                std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildrenRecursivelyExceptTypes(*layer, {"Pooling"});
                std::string concatName;
                for (const CNNLayerPtr child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                  for (const CNNLayerPtr child : children) {" << std::endl;
                    if (child->type == "Concat") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                      if (child->type == 'Concat') {" << std::endl;
                        if (!concatName.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                          if (!concatName.empty()) {" << std::endl;
                            THROW_IE_EXCEPTION << "several concat children layers are not supported";
                        }
                        concatName = child->name;
                    }
                }

                childName = concatName;
                layer = CNNNetworkHelper::getParent(*layer, 0);
            } else if (CaselessEq<std::string>()(layer->type, "Concat")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              } else if (CaselessEq<std::string>()(layer->type, 'Concat')) {" << std::endl;
                concatLayers.push_back(layer);

                if (layer->outData[0]->getInputTo().size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                  if (layer->outData[0]->getInputTo().size() != 1) {" << std::endl;
                    sideOutputLayers.push_back(layer);
                    childrenNameSideOutputLayers.push_back(childName);
                }
                int size = layer->insData.size();
                childName = layer->name;
                for (int i = 0; i < size; i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                  for (int i = 0; i < size; i++) {" << std::endl;
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
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                          childrenNameSideOutputLayers)) {" << std::endl;
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

void ConcatMultiChannelsTransformation::transform(TransformationContext& context, CNNLayer& concat) const {
    if (!canBeTransformed(context, concat)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if (!canBeTransformed(context, concat)) {" << std::endl;
        return;
    }

    if (!CaselessEq<std::string>()(concat.type, "Concat")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if (!CaselessEq<std::string>()(concat.type, 'Concat')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer type '" << concat.name << "' is not correct";
    }

    if ((concat.insData.size() < 2)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if ((concat.insData.size() < 2)) {" << std::endl;
        THROW_IE_EXCEPTION << "layer inputs '" << concat.insData.size() << "' is not correct";
    }

    if (concat.GetParamAsUInt("axis", 1) != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if (concat.GetParamAsUInt('axis', 1) != 1) {" << std::endl;
        return;
    }

    std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(concat);
    if (CNNNetworkHelper::IsChild(children, {"Concat"}, {"Pooling"})) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if (CNNNetworkHelper::IsChild(children, {'Concat'}, {'Pooling'})) {" << std::endl;
        return;
    }

    std::vector<CNNLayerPtr> quantizeLayers;
    std::vector<std::vector<CNNLayerPtr>> intermediateLayers;
    std::vector<CNNLayerPtr> concatLayers;
    std::vector<std::string> childNameOurAfterQuantizeLayers;
    std::vector<CNNLayerPtr> sideOutputLayers;
    std::vector<std::string> childrenNameSideOutputLayers;
    const auto inputDataNumber = concat.insData.size();
    for (size_t index = 0lu; index < inputDataNumber; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      for (size_t index = 0lu; index < inputDataNumber; index++) {" << std::endl;
        DataPtr quantizeOnData = concat.insData[index].lock();
        if (quantizeOnData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          if (quantizeOnData == nullptr) {" << std::endl;
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
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              childrenNameSideOutputLayers)) {" << std::endl;
            return;
        }
    }
    concatLayers.insert(concatLayers.begin(), std::make_shared<CNNLayer>(concat));

    if (quantizeLayers.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if (quantizeLayers.empty()) {" << std::endl;
        return;
    }

    if ((!isCascade(concatLayers)) || (!isMultiChannel(concatLayers))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if ((!isCascade(concatLayers)) || (!isMultiChannel(concatLayers))) {" << std::endl;
        ConcatTransformation::transform(context, concat);
        return;
    }

    // TODO: check if precisions are different and return
    std::vector<std::pair<CNNLayerPtr, std::vector<CNNLayerPtr>>> fakeQuantizeForConcatLayers;
    const DataPrecision dataPrecision = getDataPrecision(*quantizeLayers[0], QuantizationDetails::getDetails(*quantizeLayers[0]), false, false);
    if (dataPrecision.precision == Precision::UNSPECIFIED) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if (dataPrecision.precision == Precision::UNSPECIFIED) {" << std::endl;
        return;
    }

    std::vector<float> finalDequantizationScales;
    std::vector<float> finalDequantizationShifts;
    const auto parentsCount = quantizeLayers.size();

    std::unordered_map<std::string, std::vector<float>> dequantizationScalesLayers;
    std::unordered_map<std::string, std::vector<float>> dequantizationShiftsLayers;

    for (int index = (concatLayers.size() - 1); index >= 0; --index) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      for (int index = (concatLayers.size() - 1); index >= 0; --index) {" << std::endl;
        const CNNLayerPtr concatLayer = concatLayers[index];

        const std::vector<CNNLayerPtr> parents =
            CNNNetworkHelper::getParentsRecursivelyExceptTypes(*concatLayer, {"Pooling"});
        for (const CNNLayerPtr parent : parents) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          for (const CNNLayerPtr parent : parents) {" << std::endl;
            if ((parent->type != "FakeQuantize") && (parent->type != "Concat")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              if ((parent->type != 'FakeQuantize') && (parent->type != 'Concat')) {" << std::endl;
                // TODO: handle
                THROW_IE_EXCEPTION << "layer type '" << parent->type << "' not supported";
            }
        }

        for (const CNNLayerPtr fakeQuantizeLayer : parents) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          for (const CNNLayerPtr fakeQuantizeLayer : parents) {" << std::endl;
            if (fakeQuantizeLayer->type != "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              if (fakeQuantizeLayer->type != 'FakeQuantize') {" << std::endl;
                continue;
            }

            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*fakeQuantizeLayer);
            const size_t channelsCount = CNNNetworkHelper::getOutputChannelsCount(*fakeQuantizeLayer);
            std::vector<float> dequantizationScales(channelsCount);
            std::vector<float> dequantizationShifts(channelsCount);
            for (size_t i = 0ul; i < channelsCount; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              for (size_t i = 0ul; i < channelsCount; ++i) {" << std::endl;
                dequantizationScales[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
                    (quantizationDetails.outputHighValues[0] - quantizationDetails.outputLowValues[0]) / (dataPrecision.max - dataPrecision.min) :
                    1.0;

                dequantizationShifts[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
                    (quantizationDetails.outputHighValues[0] - (quantizationDetails.outputHighValues[0] - quantizationDetails.outputLowValues[0]) *
                    (dataPrecision.max / (dataPrecision.max - dataPrecision.min))) :
                    0.f;
            }
            checkAndUpdateDequantizationShiftWithZero(quantizationDetails, dequantizationShifts);

            finalDequantizationScales.insert(finalDequantizationScales.end(), dequantizationScales.begin(), dequantizationScales.end());
            finalDequantizationShifts.insert(finalDequantizationShifts.end(), dequantizationShifts.begin(), dequantizationShifts.end());

            dequantizationScalesLayers[fakeQuantizeLayer->name] = dequantizationScales;
            dequantizationShiftsLayers[fakeQuantizeLayer->name] = dequantizationShifts;

            if (QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              if (QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) {" << std::endl;
                CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 3, dataPrecision.min);
                CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 4, dataPrecision.max);

                if (updatePrecisions) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                  if (updatePrecisions) {" << std::endl;
                    CNNNetworkHelper::setOutDataPrecision(*fakeQuantizeLayer, dataPrecision.precision);

                    const std::vector<CNNLayerPtr>& intermediateLayersList = intermediateLayers[index];
                    for (const CNNLayerPtr intermediateLayer : intermediateLayersList) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                      for (const CNNLayerPtr intermediateLayer : intermediateLayersList) {" << std::endl;
                        CNNNetworkHelper::setOutDataPrecision(*intermediateLayer, dataPrecision.precision);
                    }
                }
            }
        }
    }

    // Add scaleshift at other outputs of the Quantize layer
    for (int index = 0; index < parentsCount; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      for (int index = 0; index < parentsCount; index++) {" << std::endl;
        const CNNLayer fakeQuantize = *quantizeLayers[index];
        context.quantizedFakeQuantizeNames.insert(fakeQuantize.name);

        std::vector<CNNLayerPtr> children =
            CNNNetworkHelper::getChildrenRecursivelyExceptTypes(fakeQuantize, {"Pooling"});
        for (auto it = children.begin(); it != children.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          for (auto it = children.begin(); it != children.end(); ++it) {" << std::endl;
            CNNLayerPtr child = *it;
            if (index < childNameOurAfterQuantizeLayers.size()
                    && child->name == childNameOurAfterQuantizeLayers[index]) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                      && child->name == childNameOurAfterQuantizeLayers[index]) {" << std::endl;
                children.erase(it, it + 1);
                break;
            }
        }

        for (const CNNLayerPtr child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          for (const CNNLayerPtr child : children) {" << std::endl;
            const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*child);

            auto dequantizationScalesIt = dequantizationScalesLayers.find(fakeQuantize.name);
            if (dequantizationScalesIt == dequantizationScalesLayers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              if (dequantizationScalesIt == dequantizationScalesLayers.end()) {" << std::endl;
                THROW_IE_EXCEPTION << "dequantization scales not found for layer " << fakeQuantize.name;
            }

            auto dequantizationShiftIt = dequantizationShiftsLayers.find(fakeQuantize.name);
            if (dequantizationShiftIt == dequantizationShiftsLayers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              if (dequantizationShiftIt == dequantizationShiftsLayers.end()) {" << std::endl;
                THROW_IE_EXCEPTION << "dequantization shifts not found for layer " << fakeQuantize.name;
            }

            const size_t fakeQuantizeOutputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(fakeQuantize);
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context, quantizeLayers[index], child,
                DequantizationDetails(dequantizationScalesIt->second, dequantizationShiftIt->second,
                                      fakeQuantizeOutputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);

            if (updatePrecisions &&
                QuantizationDetails::isSupportedLevel(QuantizationDetails::getDetails(fakeQuantize).levels)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:                  QuantizationDetails::isSupportedLevel(QuantizationDetails::getDetails(fakeQuantize).levels)) {" << std::endl;
                CNNNetworkHelper::setOutDataPrecision(CNNNetworkHelper::getLayers(fakeQuantize, *dequantizationLayer),
                                                      dataPrecision.precision);
            }
        }
    }

    if (updatePrecisions) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if (updatePrecisions) {" << std::endl;
        CNNNetworkHelper::setOutDataPrecision(concat, dataPrecision.precision);
        for (const CNNLayerPtr concatLayer : concatLayers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          for (const CNNLayerPtr concatLayer : concatLayers) {" << std::endl;
            if (concatLayer->name == concat.name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:              if (concatLayer->name == concat.name) {" << std::endl;
                continue;
            }

            // TODO: check if the same precision is used: U8 or S8 for all concat layers
            // TODO: fix & remove
            concatLayer->insData[0].lock()->setPrecision(dataPrecision.precision);
            // TODO: workaround
            concatLayer->precision = dataPrecision.precision;
            CNNNetworkHelper::setOutDataPrecision(*concatLayer, dataPrecision.precision);
        }
    }

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(concat);

    // Add scaleshift at outputs of our layers
    children = CNNNetworkHelper::getChildren(concat);
    if (children.size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      if (children.size() == 0) {" << std::endl;
        const std::string originalName = concat.name;
        CNNNetworkHelper::renameLayer(context.network, concat.name, concat.name + LayerTransformation::lastLayerPrefix);

        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context, std::make_shared<CNNLayer>(concat), nullptr,
            DequantizationDetails(finalDequantizationScales, finalDequantizationShifts, outputChannelsCount),
            originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          for (const CNNLayerPtr child : children) {" << std::endl;
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context, std::make_shared<CNNLayer>(concat), child,
                DequantizationDetails(finalDequantizationScales, finalDequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }

    // Add scaleshift at outputs of side branches
    for (int index = 0; index < sideOutputLayers.size(); index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:      for (int index = 0; index < sideOutputLayers.size(); index++) {" << std::endl;
        const CNNLayerPtr concatLayer = sideOutputLayers[index];

        const size_t concatOutputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*concatLayer);
        std::vector<float> dequantizationScales1(concatOutputChannelsCount);
        std::vector<float> dequantizationShifts1(concatOutputChannelsCount);
        for (size_t index_ = 0; index_ < concatOutputChannelsCount; ++index_) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          for (size_t index_ = 0; index_ < concatOutputChannelsCount; ++index_) {" << std::endl;
            dequantizationScales1[index_] = finalDequantizationScales[index_];
            dequantizationShifts1[index_] = finalDequantizationShifts[index_];
        }

        std::vector<CNNLayerPtr> children =
            CNNNetworkHelper::getChildren(*concatLayer, childrenNameSideOutputLayers[index]);
        for (int i = 0; i < children.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/concat_multi_channels.cpp:          for (int i = 0; i < children.size(); i++) {" << std::endl;
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context, std::make_shared<CNNLayer>(*sideOutputLayers[index]), children[i],
                DequantizationDetails(dequantizationScales1, dequantizationShifts1, concatOutputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}
