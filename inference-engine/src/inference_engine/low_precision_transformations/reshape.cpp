#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reshape.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

size_t getChannelVolume(const SizeVector& dims) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:  size_t getChannelVolume(const SizeVector& dims) {" << std::endl;
    size_t volume = 1ul;
    for (size_t i = 2; i < dims.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      for (size_t i = 2; i < dims.size(); ++i) {" << std::endl;
        volume = volume * dims[i];
    }

    return volume;
}

void ReshapeTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (!canBeTransformed(context, layer)) {" << std::endl;
        return;
    }

    if ((layer.insData.size() == 0) || layer.insData.size() > 2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if ((layer.insData.size() == 0) || layer.insData.size() > 2) {" << std::endl;
        THROW_IE_EXCEPTION << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Reshape")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (!CaselessEq<std::string>()(layer.type, 'Reshape')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer '" << layer.name << "' is not correct";
    }

    if (layer.insData.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (layer.insData.size() > 1) {" << std::endl;
        transformOriginal(context, layer);
    } else {
        transformConstPropagated(context, layer);
    }
}

bool ReshapeTransformation::canTransformOriginal(const CNNLayer& layer) const {
    const CNNLayerPtr constLayer = CNNNetworkHelper::getParent(layer, 1);
    if (constLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (constLayer == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer '" << layer.name << "' does not have parent at 1 position";
    }
    if (constLayer->type != "Const") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (constLayer->type != 'Const') {" << std::endl;
        return false;
    }

    const Blob::Ptr paramsBlob = CNNNetworkHelper::getBlob(constLayer, "custom");
    const Precision precision = paramsBlob->getTensorDesc().getPrecision();
    if (!CNNNetworkHelper::isBlobPrecisionSupported(precision)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (!CNNNetworkHelper::isBlobPrecisionSupported(precision)) {" << std::endl;
        THROW_IE_EXCEPTION << "layer " << constLayer->type << " '" << constLayer->name << "' unexpected precision " << precision;
    }

    if (paramsBlob->size() < 2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (paramsBlob->size() < 2) {" << std::endl;
        return false;
    }

    const DataPtr inputData = layer.insData[0].lock();
    if (inputData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (inputData == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "input data is absent";
    }

    const std::vector<size_t> inputDims = inputData->getTensorDesc().getDims();
    if (inputDims.size() < 2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (inputDims.size() < 2) {" << std::endl;
        return false;
    }

    std::shared_ptr<float> paramsBufferData = CNNNetworkHelper::getFloatData(paramsBlob);
    float* params = paramsBufferData.get();
    if (((params[0] != -1) && (params[0] != 0) && (inputDims[0] != params[0])) ||
        ((params[1] != -1) && (params[1] != 0) && (inputDims[1] != params[1]))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:          ((params[1] != -1) && (params[1] != 0) && (inputDims[1] != params[1]))) {" << std::endl;
        return false;
    }

    return true;
}

void ReshapeTransformation::transformOriginal(TransformationContext& context, CNNLayer& layer) const {
    if (!canTransformOriginal(layer)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (!canTransformOriginal(layer)) {" << std::endl;
        return;
    }

    const CNNLayerPtr constLayer = CNNNetworkHelper::getParent(layer, 1);
    const Blob::Ptr paramsBlob = CNNNetworkHelper::getBlob(constLayer, "custom");
    const signed int* paramsBuffer = paramsBlob->buffer().as<const signed int*>();
    if (paramsBuffer[1] == -1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (paramsBuffer[1] == -1) {" << std::endl;
        quantize(context, layer);
        return;
    }

    TransparentBaseTransformation::transform(context, layer);
}

bool ReshapeTransformation::canTransformConstPropagated(const CNNLayer& layer) const {
    if (layer.insData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (layer.insData.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected input count " << layer.insData.size();
    }
    const DataPtr input = layer.insData[0].lock();
    if (input == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (input == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "input is absent";
    }
    const std::vector<size_t> inputDims = input->getDims();
    if (inputDims.size() < 2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (inputDims.size() < 2) {" << std::endl;
        return false;
    }

    if (layer.outData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (layer.outData.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected output count " << layer.outData.size();
    }
    const std::vector<size_t> outputDims = layer.outData[0]->getDims();
    if (outputDims.size() < 2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (outputDims.size() < 2) {" << std::endl;
        return false;
    }

    const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::getParent(layer, 0ul);
    if ((dequantizationLayer->outData[0]->getTensorDesc().getLayout() != Layout::NCHW) || (layer.outData[0]->getTensorDesc().getLayout() != Layout::NC)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if ((dequantizationLayer->outData[0]->getTensorDesc().getLayout() != Layout::NCHW) || (layer.outData[0]->getTensorDesc().getLayout() != Layout::NC)) {" << std::endl;
        for (size_t i = 0; i < 2; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:          for (size_t i = 0; i < 2; ++i) {" << std::endl;
            if (inputDims[i] != outputDims[i]) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:              if (inputDims[i] != outputDims[i]) {" << std::endl;
                return false;
            }
        }
    }

    return true;
}

void ReshapeTransformation::transformConstPropagated(TransformationContext& context, CNNLayer& layer) const {
    if (!canTransformConstPropagated(layer)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (!canTransformConstPropagated(layer)) {" << std::endl;
        return;
    }

    const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::getParent(layer, 0ul);
    if ((dequantizationLayer->outData[0]->getTensorDesc().getLayout() == Layout::NCHW) && (layer.outData[0]->getTensorDesc().getLayout() == Layout::NC)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if ((dequantizationLayer->outData[0]->getTensorDesc().getLayout() == Layout::NCHW) && (layer.outData[0]->getTensorDesc().getLayout() == Layout::NC)) {" << std::endl;
        quantize(context, layer);
        return;
    }

    TransparentBaseTransformation::transform(context, layer);
}

void ReshapeTransformation::quantize(TransformationContext& context, CNNLayer& layer) const {
    const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::getParent(layer, 0ul);
    if ((dequantizationLayer == nullptr) || (dequantizationLayer->type != "ScaleShift")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if ((dequantizationLayer == nullptr) || (dequantizationLayer->type != 'ScaleShift')) {" << std::endl;
        return;
    }

    const size_t inputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*dequantizationLayer);
    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(layer);
    const size_t channelVolume = getChannelVolume(layer.insData[0].lock()->getTensorDesc().getDims());

    if (layer.insData[0].lock()->getTensorDesc().getDims()[0] != dequantizationLayer->insData[0].lock()->getTensorDesc().getDims()[0] ||
        inputChannelsCount * channelVolume != outputChannelsCount)
        return;

    std::vector<float> originalDataDequantizationScales;
    std::vector<float> originalDataDequantizationShifts;
    fillFromDequantizationLayer(*dequantizationLayer, originalDataDequantizationScales, originalDataDequantizationShifts);

    std::vector<float> dequantizationScales(outputChannelsCount);
    std::vector<float> dequantizationShifts(outputChannelsCount);

    for (size_t inputChannel = 0ul; inputChannel < inputChannelsCount; inputChannel++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      for (size_t inputChannel = 0ul; inputChannel < inputChannelsCount; inputChannel++) {" << std::endl;
        for (size_t i = 0ul; i < channelVolume; i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:          for (size_t i = 0ul; i < channelVolume; i++) {" << std::endl;
            dequantizationScales[inputChannel * channelVolume + i] = originalDataDequantizationScales[inputChannel];
            dequantizationShifts[inputChannel * channelVolume + i] = originalDataDequantizationShifts[inputChannel];
        }
    }

    if (updatePrecisions) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (updatePrecisions) {" << std::endl;
        const Precision lowPrecision = getPrecisionBeforeParentDequantizationScaleShift(layer);
        CNNNetworkHelper::setOutDataPrecision(layer, lowPrecision);
    }

    CNNNetworkHelper::removeLayer(context.network, dequantizationLayer);
    context.removeLayer(*dequantizationLayer);

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    if (children.size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:      if (children.size() == 0) {" << std::endl;
        const std::string originalName = layer.name;
        CNNNetworkHelper::renameLayer(context.network, layer.name, layer.name + LayerTransformation::lastLayerPrefix);

        const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context,
            std::make_shared<CNNLayer>(layer),
            nullptr,
            DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount),
            originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr& child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/reshape.cpp:          for (const CNNLayerPtr& child : children) {" << std::endl;
            const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(layer),
                child,
                DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}

bool ReshapeTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return (layer.insData.size() > 1) ? canTransformOriginal(layer) : canTransformConstPropagated(layer);
}
