#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/quantization_details.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>
#include <math.h>

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

using namespace InferenceEngine;
using namespace InferenceEngine::details;

class ConstTensorDesc {
public:
    static void validate(const Layout layout, const std::vector<size_t>& dims) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      static void validate(const Layout layout, const std::vector<size_t>& dims) {" << std::endl;
        switch (layout) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:          switch (layout) {" << std::endl;
        case Layout::SCALAR: {
            if (dims.size() != 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:              if (dims.size() != 0) {" << std::endl;
                THROW_IE_EXCEPTION << "unexpected dimensions size " << dims.size() << " for layout " << layout;
            }
            break;
        }
        case Layout::C: {
            if (dims.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:              if (dims.size() != 1) {" << std::endl;
                THROW_IE_EXCEPTION << "unexpected dimensions size " << dims.size() << " for layout " << layout;
            }
            break;
        }
        case Layout::NCHW: {
            if (dims.size() != 4) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:              if (dims.size() != 4) {" << std::endl;
                THROW_IE_EXCEPTION << "unexpected dimensions size " << dims.size() << " for layout " << layout;
            }
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "unexpected layout " << layout;
        }
        }
    }

    static size_t getChannelsCount(const Layout layout, const std::vector<size_t>& dims) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      static size_t getChannelsCount(const Layout layout, const std::vector<size_t>& dims) {" << std::endl;
        switch (layout) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:          switch (layout) {" << std::endl;
        case Layout::SCALAR: {
            return 1;
        }
        case Layout::C: {
            return dims[0];
        }
        case Layout::NCHW: {
            return dims[1];
        }
        default: {
            THROW_IE_EXCEPTION << "unexpected layout " << layout;
        }
        }
    }
};

QuantizationDetails::QuantizationDetails()
    : levels(),
      inputLowValues({}),
      inputHighValues({}),
      outputLowValues({}),
      outputHighValues({}),
      inputIntervalsCount(0),
      outputIntervalsCount(0),
      outputChannelsCount(0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:        outputChannelsCount(0) {" << std::endl;}

QuantizationDetails::QuantizationDetails(const QuantizationDetails& quantizationDetails)
    : levels(quantizationDetails.levels),
      inputLowValues(quantizationDetails.inputLowValues),
      inputHighValues(quantizationDetails.inputHighValues),
      outputLowValues(quantizationDetails.outputLowValues),
      outputHighValues(quantizationDetails.outputHighValues),
      inputIntervalsCount(quantizationDetails.inputIntervalsCount),
      outputIntervalsCount(quantizationDetails.outputIntervalsCount),
      outputChannelsCount(quantizationDetails.outputChannelsCount) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:        outputChannelsCount(quantizationDetails.outputChannelsCount) {" << std::endl;}

QuantizationDetails::QuantizationDetails(const size_t levels, const std::vector<float>& inputLowValues,
                                         const std::vector<float>& inputHighValues,
                                         const std::vector<float>& outputLowValues,
                                         const std::vector<float>& outputHighValues, const size_t inputIntervalsCount,
                                         const size_t outputIntervalsCount, const size_t outputChannelsCount)
    : levels(levels),
      inputLowValues(inputLowValues),
      inputHighValues(inputHighValues),
      outputLowValues(outputLowValues),
      outputHighValues(outputHighValues),
      inputIntervalsCount(inputIntervalsCount),
      outputIntervalsCount(outputIntervalsCount),
      outputChannelsCount(outputChannelsCount) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:        outputChannelsCount(outputChannelsCount) {" << std::endl;}

QuantizationDetails QuantizationDetails::getDetails(const CNNLayer& quantize) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:  QuantizationDetails QuantizationDetails::getDetails(const CNNLayer& quantize) {" << std::endl;
    if (quantize.insData.size() != 5) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if (quantize.insData.size() != 5) {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected inputs size " << quantize.insData.size() << " for Quantize layer '"
                           << quantize.name;
    }
    for (int i = 0; i < quantize.insData.size(); i++)
        if (quantize.insData[i].lock() == nullptr)
            THROW_IE_EXCEPTION << "Invalid input data for layer '" << quantize.name << "' with index " << i;

    if (!quantize.CheckParamPresence("levels")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if (!quantize.CheckParamPresence('levels')) {" << std::endl;
        THROW_IE_EXCEPTION << "Parameter 'levels' is absent for Quantize layer '" << quantize.name << "'";
    }
    const auto levels = quantize.GetParamAsInt("levels");

    const CNNLayerPtr inputLowLayer = quantize.insData[1].lock()->getCreatorLayer().lock();
    validate(inputLowLayer);
    const std::vector<float> inputLowValues = getBlobValue(inputLowLayer);

    const CNNLayerPtr inputHighLayer = quantize.insData[2].lock()->getCreatorLayer().lock();
    validate(inputHighLayer);
    const std::vector<float> inputHighValues = getBlobValue(inputHighLayer);

    if (inputLowValues.size() != inputHighValues.size()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if (inputLowValues.size() != inputHighValues.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Quantize input values sizes are not equal for layer " << quantize.name;
    }

    const size_t inputIntervalsCount = inputLowValues.size();

    const CNNLayerPtr outputLowLayer = quantize.insData[3].lock()->getCreatorLayer().lock();
    validate(outputLowLayer);
    const std::vector<float> outputLowValues = getBlobValue(outputLowLayer);

    const CNNLayerPtr outputHighLayer = quantize.insData[4].lock()->getCreatorLayer().lock();
    validate(outputHighLayer);
    const std::vector<float> outputHighValues = getBlobValue(outputHighLayer);

    if (outputLowValues.size() != outputHighValues.size()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if (outputLowValues.size() != outputHighValues.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Quantize output values sizes are not equal for layer " << quantize.name;
    }

    const size_t outputIntervalsCount = outputLowValues.size();

    const size_t outputChannelsCount =
        CNNNetworkHelper::getOutputChannelsCount(quantize, CNNNetworkHelper::onWeights(quantize));
    if ((outputIntervalsCount != 1) && (outputIntervalsCount != outputChannelsCount)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if ((outputIntervalsCount != 1) && (outputIntervalsCount != outputChannelsCount)) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected output channels count " << outputChannelsCount;
    }

    return QuantizationDetails(levels, inputLowValues, inputHighValues, outputLowValues, outputHighValues,
                               inputIntervalsCount, outputIntervalsCount, outputChannelsCount);
}

bool QuantizationDetails::hasNegativeOutput() const {
    for (const float value : outputLowValues) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      for (const float value : outputLowValues) {" << std::endl;
        if (value < 0.f) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:          if (value < 0.f) {" << std::endl;
            return true;
        }
    }

    for (const float value : outputHighValues) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      for (const float value : outputHighValues) {" << std::endl;
        if (value < 0.f) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:          if (value < 0.f) {" << std::endl;
            return true;
        }
    }

    return false;
}

float QuantizationDetails::maxOutput(const size_t channel) const {
    const auto value = fmax(fabs(outputLowValues[outputLowValues.size() == 1 ? 0 : channel]),
                            fabs(outputHighValues[outputHighValues.size() == 1 ? 0 : channel]));
    return value;
}

float QuantizationDetails::maxInput(const size_t channel) const {
    const auto value = fmax(fabs(outputLowValues[inputLowValues.size() == 1 ? 0 : channel]),
                            fabs(outputHighValues[inputHighValues.size() == 1 ? 0 : channel]));
    return value;
}

float QuantizationDetails::maxOutputHigh() const {
    float output = getOutputHighValue(0);
    for (size_t channel = 1; channel < outputIntervalsCount; ++channel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      for (size_t channel = 1; channel < outputIntervalsCount; ++channel) {" << std::endl;
        if (output < getOutputHighValue(channel)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:          if (output < getOutputHighValue(channel)) {" << std::endl;
            output = getOutputHighValue(channel);
        }
    }
    return output;
}

float QuantizationDetails::minOutputLow() const {
    float output = getOutputLowValue(0);
    for (size_t channel = 1; channel < outputIntervalsCount; ++channel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      for (size_t channel = 1; channel < outputIntervalsCount; ++channel) {" << std::endl;
        if (output > getOutputLowValue(channel)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:          if (output > getOutputLowValue(channel)) {" << std::endl;
            output = getOutputLowValue(channel);
        }
    }
    return output;
}

float QuantizationDetails::getInputLowValue(const size_t channel) const {
    if ((inputIntervalsCount != 1) && (channel >= inputIntervalsCount)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if ((inputIntervalsCount != 1) && (channel >= inputIntervalsCount)) {" << std::endl;
        THROW_IE_EXCEPTION << "channel " << channel << " is out of bound, input channels count " << inputIntervalsCount;
    }
    const float value = inputLowValues.size() == 1 ? inputLowValues[0] : inputLowValues[channel];
    return value;
}

float QuantizationDetails::getInputHighValue(const size_t channel) const {
    if ((inputIntervalsCount != 1) && (channel >= inputIntervalsCount)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if ((inputIntervalsCount != 1) && (channel >= inputIntervalsCount)) {" << std::endl;
        THROW_IE_EXCEPTION << "channel " << channel << " is out of bound, input channels count " << inputIntervalsCount;
    }
    const float value = inputHighValues.size() == 1 ? inputHighValues[0] : inputHighValues[channel];
    return value;
}

float QuantizationDetails::getOutputLowValue(const size_t channel) const {
    if ((outputIntervalsCount != 1) && (channel >= outputIntervalsCount)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if ((outputIntervalsCount != 1) && (channel >= outputIntervalsCount)) {" << std::endl;
        THROW_IE_EXCEPTION << "channel " << channel << " is out of bound, output channels count "
                           << outputIntervalsCount;
    }
    const float value = outputLowValues.size() == 1 ? outputLowValues[0] : outputLowValues[channel];
    return value;
}

float QuantizationDetails::getOutputHighValue(const size_t channel) const {
    if ((outputIntervalsCount != 1) && (channel >= outputIntervalsCount)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if ((outputIntervalsCount != 1) && (channel >= outputIntervalsCount)) {" << std::endl;
        THROW_IE_EXCEPTION << "channel " << channel << " is out of bound, output channels count "
                           << outputIntervalsCount;
    }
    const float value = outputHighValues.size() == 1 ? outputHighValues[0] : outputHighValues[channel];
    return value;
}

void QuantizationDetails::validate(const CNNLayerPtr& constantLayer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:  void QuantizationDetails::validate(const CNNLayerPtr& constantLayer) {" << std::endl;
    if (constantLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if (constantLayer == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "Quantize layer input is absent";
    }

    if (constantLayer->blobs.size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if (constantLayer->blobs.size() == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' doesn't have blobs";
    }

    if (constantLayer->blobs.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      if (constantLayer->blobs.size() > 1) {" << std::endl;
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' has too much blobs";
    }

    const auto blob = constantLayer->blobs.begin()->second;
    // const auto byteSize = blob->byteSize();
    // if ((blob->getTensorDesc().getDims().size() != 0) &&
    //     (blob->getTensorDesc().getDims().size() != 1) &&
    //     (blob->getTensorDesc().getDims().size() != 4)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      //     (blob->getTensorDesc().getDims().size() != 4)) {" << std::endl;
    //     THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' blob dimensions are not correct";
    // }

    const auto tensorDesc = blob->getTensorDesc();
    // if ((tensorDesc.getLayout() != Layout::SCALAR) &&
    //     (tensorDesc.getLayout() != Layout::C) &&
    //     ((tensorDesc.getLayout() != Layout::NCHW))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      //     ((tensorDesc.getLayout() != Layout::NCHW))) {" << std::endl;
    //     THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' layout not correct";
    // }

    // const auto dims = tensorDesc.getDims();
    // if ((dims.size() != 0) && (dims.size() != 1) && (dims.size() != 4)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:      // if ((dims.size() != 0) && (dims.size() != 1) && (dims.size() != 4)) {" << std::endl;
    //     THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' blob dimensions size " <<
    //     dims.size() << " not correct";
    // }

    // ConstTensorDesc::validate(tensorDesc.getLayout(), tensorDesc.getDims());
}

std::vector<float> QuantizationDetails::getBlobValue(const CNNLayerPtr& constantLayer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:  std::vector<float> QuantizationDetails::getBlobValue(const CNNLayerPtr& constantLayer) {" << std::endl;
    const auto blob = constantLayer->blobs.begin()->second;
    auto buffer = CNNNetworkHelper::getFloatData(blob);
    return std::vector<float>(buffer.get(), buffer.get() + blob->size());
}

bool QuantizationDetails::isSupportedLevel(const size_t level) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/quantization_details.cpp:  bool QuantizationDetails::isSupportedLevel(const size_t level) {" << std::endl;
    static const std::unordered_set<size_t> supported_levels = { 15ul, 16ul, 255ul, 256ul };
    return supported_levels.find(level) != supported_levels.end();
}
