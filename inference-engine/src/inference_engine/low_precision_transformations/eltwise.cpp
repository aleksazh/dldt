#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/eltwise.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>

#include <algorithm>
#include <details/caseless.hpp>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ie_util_internal.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool EltwiseTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if (!LayerTransformation::canBeTransformed(context, layer)) {" << std::endl;
        return false;
    }

    if (!CaselessEq<std::string>()(layer.type, "Eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if (!CaselessEq<std::string>()(layer.type, 'Eltwise')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer type '" << layer.name << "' is not correct";
    }

    const TensorDesc& tensorDesc0 = layer.insData[0].lock()->getTensorDesc();
    for (size_t i = 1ul; i < layer.insData.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      for (size_t i = 1ul; i < layer.insData.size(); ++i) {" << std::endl;
        const auto& data = layer.insData[i];
        if (!isSupported(tensorDesc0, data.lock()->getTensorDesc())) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if (!isSupported(tensorDesc0, data.lock()->getTensorDesc())) {" << std::endl;
            return false;
        }
    }

    return true;
}

void EltwiseTransformation::transform(TransformationContext& context, CNNLayer& eltwise) const {
    if (!canBeTransformed(context, eltwise)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if (!canBeTransformed(context, eltwise)) {" << std::endl;
        return;
    }

    if ((!eltwise.CheckParamPresence("operation")) || (eltwise.GetParamAsString("operation") != "sum")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if ((!eltwise.CheckParamPresence('operation')) || (eltwise.GetParamAsString('operation') != 'sum')) {" << std::endl;
        return;
    }

    if (!CaselessEq<std::string>()(eltwise.type, "Eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if (!CaselessEq<std::string>()(eltwise.type, 'Eltwise')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer type '" << eltwise.name << "' is not correct";
    }

    if ((eltwise.insData.size() < 2) || (!eltwise.CheckParamPresence("operation")) ||
        (eltwise.GetParamAsString("operation") != "sum")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          (eltwise.GetParamAsString('operation') != 'sum')) {" << std::endl;
        return;
    }

    std::vector<CNNLayerPtr> quantizeLayers;
    const size_t numberInputLayers = eltwise.insData.size();
    const size_t outputChannelCount = CNNNetworkHelper::getOutputChannelsCount(eltwise);

    std::vector<std::string> childNameOurAfterQuantizeLayers;
    childNameOurAfterQuantizeLayers.resize(numberInputLayers);

    size_t quantizationLevels = 0lu;
    std::vector<QuantizationDetails> quantizationLayersDetails;
    for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
        DataPtr quantizeOnData = eltwise.insData[index].lock();
        if (quantizeOnData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if (quantizeOnData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input is absent";
        }

        auto layer = quantizeOnData->getCreatorLayer().lock();
        if ((layer->type != "FakeQuantize") && (layer->type != "Quantize")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if ((layer->type != 'FakeQuantize') && (layer->type != 'Quantize')) {" << std::endl;
            do {
                if (layer->type == "Pooling") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:                  if (layer->type == 'Pooling') {" << std::endl;
                    childNameOurAfterQuantizeLayers[index] = layer->name;
                    layer = CNNNetworkHelper::getParent(*layer, 0);
                } else {
                    return;
                }
            } while ((layer->type != "FakeQuantize") && (layer->type != "Quantize"));
        } else {
            childNameOurAfterQuantizeLayers[index] = eltwise.name;
        }

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(*layer);
        if (!QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) continue;
        if (quantizationLevels == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if (quantizationLevels == 0) {" << std::endl;
            quantizationLevels = quantizationDetails.levels;
        } else if (quantizationLevels != quantizationDetails.levels) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          } else if (quantizationLevels != quantizationDetails.levels) {" << std::endl;
            THROW_IE_EXCEPTION << "different quantization levels " << quantizationLevels << " are not supported";
        }

        quantizeLayers.push_back(layer);
        quantizationLayersDetails.push_back(quantizationDetails);
    }

    if (quantizeLayers.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if (quantizeLayers.empty()) {" << std::endl;
        return;
    }

    const DataPrecision dataPrecision = getDataPrecision(*quantizeLayers[0], QuantizationDetails::getDetails(*quantizeLayers[0]), false, false);
    if (dataPrecision.precision == Precision::UNSPECIFIED) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if (dataPrecision.precision == Precision::UNSPECIFIED) {" << std::endl;
        return;
    }

    std::vector<float> dequantizationScales(outputChannelCount);
    std::vector<float> dequantizationShifts(outputChannelCount);

    std::vector<std::vector<float>> dequantizationShiftsLayers;
    dequantizationShiftsLayers.resize(numberInputLayers);

    // TODO: refactor: use cycle anyway
    // TODO: hardcode detected: zero element

    if ((quantizationLayersDetails[0].outputHighValues.size() == 1)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if ((quantizationLayersDetails[0].outputHighValues.size() == 1)) {" << std::endl;
        std::vector<float> outputInterval;
        std::vector<float> lowNewOutput;
        float sumLowOldOutput = 0.f;
        for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
            const float outputLowValue = quantizationLayersDetails[index].outputLowValues[0];
            const float outputHighValue = quantizationLayersDetails[index].outputHighValues[0];
            outputInterval.push_back((outputHighValue - outputLowValue));
            sumLowOldOutput += outputLowValue;
        }

        if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {" << std::endl;
            const size_t minLevels = getMinQuantizationLevels(dataPrecision, outputInterval);
            if (minLevels < this->minQuantizationLevels) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              if (minLevels < this->minQuantizationLevels) {" << std::endl;
                return;
            }
        }

        const float maxOutputInterval = *std::max_element(outputInterval.begin(), outputInterval.end());
        if (maxOutputInterval == 0.f)
            THROW_IE_EXCEPTION << "Invalid output interval: " << maxOutputInterval;

        if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {" << std::endl;
            for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
                const int outputIntervalMax =
                    static_cast<int>(roundf(dataPrecision.max * outputInterval[index] / maxOutputInterval));
                if (outputIntervalMax <= INTERVALS_THRESHOLD) return;
            }
        }

        for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
            if (quantizeLayers[index] == nullptr)
                continue;
            CNNLayer& fakeQuantizeLayer = *quantizeLayers[index];

            // TODO: copy/paste, refactor: extract to MultiBranchTransformation::updateQuantizationRange
            switch (quantizedTensorAlignmentOnActivations) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              switch (quantizedTensorAlignmentOnActivations) {" << std::endl;
            case QuantizedTensorAlignment::None: {
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3,
                                              dataPrecision.min * outputInterval[index] / maxOutputInterval);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4,
                                              dataPrecision.max * outputInterval[index] / maxOutputInterval);
                break;
            }
            case QuantizedTensorAlignment::UpdateIntervals: {
                const float k = maxOutputInterval / outputInterval[index];
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 2,
                                              quantizationLayersDetails[index].inputHighValues[0] * k);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3,
                                              dataPrecision.min * outputInterval[index] / maxOutputInterval);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, dataPrecision.max);
                break;
            }
            case QuantizedTensorAlignment::UpdateLevel: {
                const float outputIntervalMin = roundf(dataPrecision.min * outputInterval[index] / maxOutputInterval);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, outputIntervalMin);

                const float outputIntervalMax = roundf(dataPrecision.max * outputInterval[index] / maxOutputInterval);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, outputIntervalMax);

                const size_t levels = static_cast<size_t>(fabs(outputIntervalMax - outputIntervalMin)) + 1ul;
                fakeQuantizeLayer.params["levels"] = std::to_string(levels);
                QuantizeLayer* layer = dynamic_cast<QuantizeLayer*>(&fakeQuantizeLayer);
                if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:                  if (layer == nullptr) {" << std::endl;
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
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              if (updatePrecisions) {" << std::endl;
                CNNNetworkHelper::setOutDataPrecision(fakeQuantizeLayer, dataPrecision.precision);
            }

            lowNewOutput.push_back(dataPrecision.min * outputInterval[index] / maxOutputInterval);

            context.quantizedFakeQuantizeNames.insert(quantizeLayers[index]->name);
        }

        float generalScaleDequantize = maxOutputInterval / (dataPrecision.max - dataPrecision.min);
        const float quantizationShift =
            (sumLowOldOutput - generalScaleDequantize * accumulate(lowNewOutput.begin(), lowNewOutput.end(), 0.0)) * -1;

        for (size_t channel = 0; channel < outputChannelCount; ++channel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          for (size_t channel = 0; channel < outputChannelCount; ++channel) {" << std::endl;
            dequantizationScales[channel] = generalScaleDequantize;
            dequantizationShifts[channel] = -1 * quantizationShift;
            for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
                dequantizationShiftsLayers[index].push_back((quantizationLayersDetails[index].outputLowValues[0] -
                                                             generalScaleDequantize * lowNewOutput[index]));
            }
        }
    } else {
        for (size_t channel = 0; channel < outputChannelCount; ++channel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          for (size_t channel = 0; channel < outputChannelCount; ++channel) {" << std::endl;
            std::vector<float> outputInterval;
            std::vector<float> lowNewOutput;
            float sumLowOldOutput = 0;
            for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
                const float outputLowValue = quantizationLayersDetails[index].getOutputLowValue(channel);
                const float outputHighValue = quantizationLayersDetails[index].getOutputHighValue(channel);
                outputInterval.push_back((outputHighValue - outputLowValue));
                sumLowOldOutput += outputLowValue;
            }

            if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {" << std::endl;
                const size_t minLevels = getMinQuantizationLevels(dataPrecision, outputInterval);
                if (minLevels < this->minQuantizationLevels) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:                  if (minLevels < this->minQuantizationLevels) {" << std::endl;
                    return;
                }
            }

            const float maxOutputInterval = *max_element(outputInterval.begin(), outputInterval.end());
            if (maxOutputInterval == 0.f)
                THROW_IE_EXCEPTION << "Invalid output interval: " << maxOutputInterval;

            if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {" << std::endl;
                for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:                  for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
                    const int outputIntervalMax =
                        static_cast<int>(roundf(dataPrecision.max * outputInterval[index] / maxOutputInterval));
                    if (outputIntervalMax <= INTERVALS_THRESHOLD) return;
                }
            }

            for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
                CNNLayer& fakeQuantizeLayer = *quantizeLayers[index];

                // TODO: copy/paste, refactor: extract to MultiBranchTransformation::updateQuantizationRange
                switch (quantizedTensorAlignmentOnActivations) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:                  switch (quantizedTensorAlignmentOnActivations) {" << std::endl;
                case QuantizedTensorAlignment::None: {
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3,
                                                  dataPrecision.min * outputInterval[index] / maxOutputInterval);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4,
                                                  dataPrecision.max * outputInterval[index] / maxOutputInterval);
                    break;
                }
                case QuantizedTensorAlignment::UpdateIntervals: {
                    const float k = maxOutputInterval / outputInterval[index];
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 2,
                                                  quantizationLayersDetails[index].inputHighValues[0] * k);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3,
                                                  dataPrecision.min * outputInterval[index] / maxOutputInterval);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, dataPrecision.max);
                    break;
                }
                case QuantizedTensorAlignment::UpdateLevel: {
                    const float outputIntervalMin = roundf(dataPrecision.min * outputInterval[index] / maxOutputInterval);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, outputIntervalMin);

                    const float outputIntervalMax = roundf(dataPrecision.max * outputInterval[index] / maxOutputInterval);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, outputIntervalMax);

                    const size_t levels = static_cast<size_t>(fabs(outputIntervalMax - outputIntervalMin)) + 1ul;
                    fakeQuantizeLayer.params["levels"] = std::to_string(levels);
                    QuantizeLayer* layer = dynamic_cast<QuantizeLayer*>(&fakeQuantizeLayer);
                    if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:                      if (layer == nullptr) {" << std::endl;
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
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:                  if (updatePrecisions) {" << std::endl;
                    CNNNetworkHelper::setOutDataPrecision(fakeQuantizeLayer, dataPrecision.precision);
                }

                lowNewOutput.push_back(dataPrecision.min * outputInterval[index] / maxOutputInterval);

                context.quantizedFakeQuantizeNames.insert(quantizeLayers[index]->name);
            }

            float generalScaleDequantize = maxOutputInterval / (dataPrecision.max - dataPrecision.min);
            const float quantizationShift =
                (sumLowOldOutput - generalScaleDequantize * accumulate(lowNewOutput.begin(), lowNewOutput.end(), 0.0)) *
                -1;
            dequantizationScales[channel] = generalScaleDequantize;
            dequantizationShifts[channel] = -1 * quantizationShift;
            for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
                dequantizationShiftsLayers[index].push_back((quantizationLayersDetails[index].outputLowValues[0] -
                                                             generalScaleDequantize * lowNewOutput[index]));
            }
        }
    }

    for (int index = 0; index < numberInputLayers; index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      for (int index = 0; index < numberInputLayers; index++) {" << std::endl;
        if (quantizeLayers[index]->outData[0]->getInputTo().size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if (quantizeLayers[index]->outData[0]->getInputTo().size() != 1) {" << std::endl;
            std::vector<CNNLayerPtr> children =
                CNNNetworkHelper::getChildren(*quantizeLayers[index], childNameOurAfterQuantizeLayers[index]);
            for (int i = 0; i < children.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              for (int i = 0; i < children.size(); i++) {" << std::endl;
                const size_t outputChannelsCount = CNNNetworkHelper::getInputChannelsCount(*quantizeLayers[index]);
                CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                    context, std::make_shared<CNNLayer>(*quantizeLayers[index]), children[i],
                    DequantizationDetails(dequantizationScales, dequantizationShiftsLayers[index],
                                          outputChannelsCount));
                context.dequantizationLayersNames.insert(dequantizationLayer->name);
            }
        }
    }
    // Add scaleshift at other outputs of the Quantize layer

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(eltwise);
    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(eltwise);
    if (children.size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if (children.size() == 0) {" << std::endl;
        const std::string originalName = eltwise.name;
        CNNNetworkHelper::renameLayer(context.network, eltwise.name,
                                      eltwise.name + LayerTransformation::lastLayerPrefix);

        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context, std::make_shared<CNNLayer>(eltwise), nullptr,
            DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount), originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr& child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          for (const CNNLayerPtr& child : children) {" << std::endl;
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context, std::make_shared<CNNLayer>(eltwise), child,
                DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}

bool EltwiseTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return false;
}

bool EltwiseTransformation::isBroadcasted(const TensorDesc& tensorDesc) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:  bool EltwiseTransformation::isBroadcasted(const TensorDesc& tensorDesc) {" << std::endl;
    const std::vector<size_t> dims = tensorDesc.getDims();
    const size_t channelIndex = dims.size() == 1 ? 0ul : (dims.size() == 2ul ? 1ul : 2ul);
    for (size_t dimension = channelIndex; dimension < dims.size(); ++dimension) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      for (size_t dimension = channelIndex; dimension < dims.size(); ++dimension) {" << std::endl;
        if (dims[dimension] != 1ul) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if (dims[dimension] != 1ul) {" << std::endl;
            return false;
        }
    }

    return true;
}

bool EltwiseTransformation::isSupported(const TensorDesc& tensorDesc1, const TensorDesc& tensorDesc2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:  bool EltwiseTransformation::isSupported(const TensorDesc& tensorDesc1, const TensorDesc& tensorDesc2) {" << std::endl;
    if (tensorDesc1.getPrecision() != tensorDesc2.getPrecision()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if (tensorDesc1.getPrecision() != tensorDesc2.getPrecision()) {" << std::endl;
        return false;
    }

    const std::vector<size_t> dims1 = tensorDesc1.getDims();
    const size_t channelsCount1 = dims1.size() == 1ul ? dims1[0] : dims1[1];
    const std::vector<size_t> dims2 = tensorDesc2.getDims();
    const size_t channelsCount2 = dims2.size() == 1ul ? dims2[0] : dims2[1];
    if ((channelsCount1 != channelsCount2) && (channelsCount1 != 1ul) && (channelsCount2 != 1ul)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if ((channelsCount1 != channelsCount2) && (channelsCount1 != 1ul) && (channelsCount2 != 1ul)) {" << std::endl;
        return false;
    }

    if (((dims1.size() == 2ul) && (channelsCount1 == 1ul)) ||
        ((dims2.size() == 2ul) && (channelsCount2 == 1ul))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          ((dims2.size() == 2ul) && (channelsCount2 == 1ul))) {" << std::endl;
        return true;
    }

    if ((dims1 == dims2) && (tensorDesc1.getLayout() != tensorDesc2.getLayout())) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if ((dims1 == dims2) && (tensorDesc1.getLayout() != tensorDesc2.getLayout())) {" << std::endl;
        return false;
    }

    if (dims1 == dims2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if (dims1 == dims2) {" << std::endl;
        return true;
    }

    if ((dims1.size() > 1ul) && (dims2.size() > 1ul)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      if ((dims1.size() > 1ul) && (dims2.size() > 1ul)) {" << std::endl;
        if (dims1[1] != dims2[1]) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if (dims1[1] != dims2[1]) {" << std::endl;
            return false;
        }

        const size_t dimensionsSize = std::min(dims1.size(), dims2.size());
        for (size_t dimension = 2ul; dimension < dimensionsSize; ++dimension) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          for (size_t dimension = 2ul; dimension < dimensionsSize; ++dimension) {" << std::endl;
            if ((dims1[dimension] != dims2[dimension]) && (dims1[dimension] != 1ul) && (dims2[dimension] != 1ul)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:              if ((dims1[dimension] != dims2[dimension]) && (dims1[dimension] != 1ul) && (dims2[dimension] != 1ul)) {" << std::endl;
                return false;
            }
        }
    }

    return true;
}

size_t EltwiseTransformation::getMinQuantizationLevels(const DataPrecision& dataPrecision, const std::vector<float>& outputIntervals) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:  size_t EltwiseTransformation::getMinQuantizationLevels(const DataPrecision& dataPrecision, const std::vector<float>& outputIntervals) {" << std::endl;
    size_t minLevels = std::numeric_limits<std::size_t>::max();
    const float maxOutputInterval = *std::max_element(outputIntervals.begin(), outputIntervals.end());
    for (int index = 0; index < outputIntervals.size(); index++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:      for (int index = 0; index < outputIntervals.size(); index++) {" << std::endl;
        const float outputIntervalMin = roundf(dataPrecision.min * outputIntervals[index] / maxOutputInterval);
        const float outputIntervalMax = roundf(dataPrecision.max * outputIntervals[index] / maxOutputInterval);
        const size_t levels = static_cast<size_t>(fabs(outputIntervalMax - outputIntervalMin)) + 1ul;
        if (minLevels > levels) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/eltwise.cpp:          if (minLevels > levels) {" << std::endl;
            minLevels = levels;
        }
    }
    return minLevels;
}
