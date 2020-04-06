#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/network_helper.hpp"

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>
#include <precision_utils.h>

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

CNNLayerPtr CNNNetworkHelper::getLayer(const ICNNNetwork& network, const std::string& layerName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  CNNLayerPtr CNNNetworkHelper::getLayer(const ICNNNetwork& network, const std::string& layerName) {" << std::endl;
    std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (CNNLayerPtr layer : layers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (CNNLayerPtr layer : layers) {" << std::endl;
        if (layer->name == layerName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (layer->name == layerName) {" << std::endl;
            return layer;
        }
    }

    return nullptr;
}

Blob::Ptr CNNNetworkHelper::makeNewBlobPtr(const TensorDesc& desc) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  Blob::Ptr CNNNetworkHelper::makeNewBlobPtr(const TensorDesc& desc) {" << std::endl;
    Blob::Ptr newBlob;
    if (desc.getPrecision() == Precision::FP32)
        newBlob = make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(desc);
    else if (desc.getPrecision() == Precision::FP16)
        newBlob = make_shared_blob<PrecisionTrait<Precision::FP16>::value_type>(desc);
    else if (desc.getPrecision() == Precision::I8)
        newBlob = make_shared_blob<PrecisionTrait<Precision::I8>::value_type>(desc);
    else if (desc.getPrecision() == Precision::U8)
        newBlob = make_shared_blob<PrecisionTrait<Precision::U8>::value_type>(desc);
    else if (desc.getPrecision() == Precision::I32)
        newBlob = make_shared_blob<PrecisionTrait<Precision::I32>::value_type>(desc);
    else
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << desc.getPrecision();

    return newBlob;
}

void CNNNetworkHelper::updateBlobs(CNNLayer& layer, const std::string& blobName, float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::updateBlobs(CNNLayer& layer, const std::string& blobName, float value) {" << std::endl;
    const auto existingBlobIt = layer.blobs.find(blobName);
    if (existingBlobIt == layer.blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (existingBlobIt == layer.blobs.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "blob '" << blobName << "' was not found in layer " << layer.name;
    }
    const auto& existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    Blob::Ptr newBlob = makeNewBlobPtr(existingBlobTensorDesc);

    newBlob->allocate();
    fillBlobByFP32(newBlob, value);
    layer.blobs[existingBlobIt->first] = newBlob;
}

void CNNNetworkHelper::invertFakeQuantize(const CNNLayer& fakeQuantize) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::invertFakeQuantize(const CNNLayer& fakeQuantize) {" << std::endl;
    if (fakeQuantize.type != "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (fakeQuantize.type != 'FakeQuantize') {" << std::endl;
        THROW_IE_EXCEPTION << "invalid layer type " << fakeQuantize.type;
    }
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);
    const size_t valuesCount =
        std::max(quantizationDetails.inputLowValues.size(), quantizationDetails.outputLowValues.size());
    std::vector<float> inputLowValues(valuesCount);
    std::vector<float> inputHightValues(valuesCount);
    std::vector<float> outputLowValues(valuesCount);
    std::vector<float> outputHighValues(valuesCount);
    bool wasInverted = false;
    for (size_t i = 0ul; i < valuesCount; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (size_t i = 0ul; i < valuesCount; ++i) {" << std::endl;
        if ((quantizationDetails.getInputLowValue(i) > quantizationDetails.getInputHighValue(i)) &&
            (quantizationDetails.getOutputLowValue(i) > quantizationDetails.getOutputHighValue(i))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              (quantizationDetails.getOutputLowValue(i) > quantizationDetails.getOutputHighValue(i))) {" << std::endl;
            inputLowValues[i] = quantizationDetails.getInputHighValue(i);
            inputHightValues[i] = quantizationDetails.getInputLowValue(i);
            outputLowValues[i] = quantizationDetails.getOutputHighValue(i);
            outputHighValues[i] = quantizationDetails.getOutputLowValue(i);
            wasInverted = true;
        } else {
            inputLowValues[i] = quantizationDetails.getInputLowValue(i);
            inputHightValues[i] = quantizationDetails.getInputHighValue(i);
            outputLowValues[i] = quantizationDetails.getOutputLowValue(i);
            outputHighValues[i] = quantizationDetails.getOutputHighValue(i);
        }
    }

    if (wasInverted) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (wasInverted) {" << std::endl;
        CNNNetworkHelper::updateBlobs(fakeQuantize, 1, inputLowValues);
        CNNNetworkHelper::updateBlobs(fakeQuantize, 2, inputHightValues);
        CNNNetworkHelper::updateBlobs(fakeQuantize, 3, outputLowValues);
        CNNNetworkHelper::updateBlobs(fakeQuantize, 4, outputHighValues);
    }
}
void CNNNetworkHelper::updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex,
                                   const std::vector<float>& values) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                     const std::vector<float>& values) {" << std::endl;
    CNNLayerPtr blobLayer = CNNNetworkHelper::getParent(quantizeLayer, constLayerIndex);
    if (blobLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (blobLayer == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "layer is absent";
    }

    const auto existingBlobIt = blobLayer->blobs.find("custom");
    if (existingBlobIt == blobLayer->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (existingBlobIt == blobLayer->blobs.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "custom blob was not found ";
    }

    TensorDesc newBlobTensorDesc;

    const TensorDesc existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    if ((existingBlobIt->second->size() != values.size()) && (values.size() != 1)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if ((existingBlobIt->second->size() != values.size()) && (values.size() != 1)) {" << std::endl;
        if (existingBlobTensorDesc.getLayout() == Layout::SCALAR) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (existingBlobTensorDesc.getLayout() == Layout::SCALAR) {" << std::endl;
            //
        } else if (existingBlobTensorDesc.getLayout() == Layout::C) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          } else if (existingBlobTensorDesc.getLayout() == Layout::C) {" << std::endl;
            if (existingBlobTensorDesc.getDims().size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (existingBlobTensorDesc.getDims().size() != 1) {" << std::endl;
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            if (existingBlobTensorDesc.getDims()[0] != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (existingBlobTensorDesc.getDims()[0] != 1) {" << std::endl;
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        } else if (existingBlobTensorDesc.getLayout() == Layout::NCHW) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          } else if (existingBlobTensorDesc.getLayout() == Layout::NCHW) {" << std::endl;
            if (existingBlobTensorDesc.getDims().size() != 4) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (existingBlobTensorDesc.getDims().size() != 4) {" << std::endl;
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            // OIHW
            if (existingBlobTensorDesc.getDims()[0] != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (existingBlobTensorDesc.getDims()[0] != 1) {" << std::endl;
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        }

        const std::vector<size_t> dims = {values.size()};
        const Layout layout = Layout::C;
        newBlobTensorDesc = TensorDesc(existingBlobTensorDesc.getPrecision(), dims, layout);
        for (DataPtr data : blobLayer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (DataPtr data : blobLayer->outData) {" << std::endl;
            data->reshape(dims, layout);
        }
    } else {
        newBlobTensorDesc = existingBlobTensorDesc;
    }

    Blob::Ptr newBlob = makeNewBlobPtr(newBlobTensorDesc);
    newBlob->allocate();
    blobLayer->blobs[existingBlobIt->first] = newBlob;

    if (values.size() == 1)
        fillBlobByFP32(newBlob, values[0]);
    else
        fillBlobByFP32(newBlob, values.data());
}

void CNNNetworkHelper::updateBlobs(CNNLayer& layer, const std::string& blobName, const std::vector<float>& values) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::updateBlobs(CNNLayer& layer, const std::string& blobName, const std::vector<float>& values) {" << std::endl;
    const auto existingBlobIt = layer.blobs.find(blobName);
    if (existingBlobIt == layer.blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (existingBlobIt == layer.blobs.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "custom blob was not found ";
    }

    TensorDesc newBlobTensorDesc;

    const TensorDesc existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    if ((existingBlobIt->second->size() != values.size()) && (values.size() != 1)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if ((existingBlobIt->second->size() != values.size()) && (values.size() != 1)) {" << std::endl;
        if (existingBlobTensorDesc.getLayout() == Layout::SCALAR) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (existingBlobTensorDesc.getLayout() == Layout::SCALAR) {" << std::endl;
            //
        } else if (existingBlobTensorDesc.getLayout() == Layout::C) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          } else if (existingBlobTensorDesc.getLayout() == Layout::C) {" << std::endl;
            if (existingBlobTensorDesc.getDims().size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (existingBlobTensorDesc.getDims().size() != 1) {" << std::endl;
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            if (existingBlobTensorDesc.getDims()[0] != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (existingBlobTensorDesc.getDims()[0] != 1) {" << std::endl;
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        } else if (existingBlobTensorDesc.getLayout() == Layout::NCHW) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          } else if (existingBlobTensorDesc.getLayout() == Layout::NCHW) {" << std::endl;
            if (existingBlobTensorDesc.getDims().size() != 4) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (existingBlobTensorDesc.getDims().size() != 4) {" << std::endl;
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            // OIHW
            if (existingBlobTensorDesc.getDims()[0] != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (existingBlobTensorDesc.getDims()[0] != 1) {" << std::endl;
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        }

        const std::vector<size_t> dims = {values.size()};
        const Layout layout = Layout::C;
        newBlobTensorDesc = TensorDesc(existingBlobTensorDesc.getPrecision(), dims, layout);
        for (DataPtr data : layer.outData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (DataPtr data : layer.outData) {" << std::endl;
            data->reshape(dims, layout);
        }
    } else {
        newBlobTensorDesc = existingBlobTensorDesc;
    }

    Blob::Ptr newBlob = makeNewBlobPtr(newBlobTensorDesc);
    newBlob->allocate();
    layer.blobs[existingBlobIt->first] = newBlob;

    if ((blobName == "weights") || (blobName == "biases")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if ((blobName == 'weights') || (blobName == 'biases')) {" << std::endl;
        WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(&layer);
        if (weightableLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (weightableLayer == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "layer '" << layer.name << "' with blob name '" << blobName << "' is not weightable";
        }
        if (blobName == "weights") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (blobName == 'weights') {" << std::endl;
            weightableLayer->_weights = newBlob;
        } else if (blobName == "biases") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          } else if (blobName == 'biases') {" << std::endl;
            weightableLayer->_biases = newBlob;
        } else {
            THROW_IE_EXCEPTION << "unexpected blob name '" << blobName << "' for layer " << layer.name;
        }
    }

    if (values.size() == 1)
        fillBlobByFP32(newBlob, values[0]);
    else
        fillBlobByFP32(newBlob, values.data());
}

void CNNNetworkHelper::updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex, float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex, float value) {" << std::endl;
    auto inData = quantizeLayer.insData[constLayerIndex].lock();
    if (inData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (inData == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "data is absent";
    }

    CNNLayerPtr blobLayer = inData->getCreatorLayer().lock();
    if (blobLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (blobLayer == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "layer is absent";
    }

    if (blobLayer->blobs.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (blobLayer->blobs.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected blobs size";
    }

    const auto existingBlobIt = blobLayer->blobs.begin();
    const auto& existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    Blob::Ptr newBlob = makeNewBlobPtr(existingBlobTensorDesc);

    newBlob->allocate();
    fillBlobByFP32(newBlob, value);
    blobLayer->blobs[existingBlobIt->first] = newBlob;
}

int CNNNetworkHelper::onWeightsInDepth(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  int CNNNetworkHelper::onWeightsInDepth(const CNNLayer& layer) {" << std::endl;
    const std::vector<CNNLayerPtr> children = getChildren(layer);
    for (const CNNLayerPtr& child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (const CNNLayerPtr& child : children) {" << std::endl;
        if ((CaselessEq<std::string>()(child->type, "Convolution") ||
            CaselessEq<std::string>()(child->type, "FullyConnected") ||
            CaselessEq<std::string>()(child->type, "GEMM")) &&
            (child->insData.size() >= 2lu)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              (child->insData.size() >= 2lu)) {" << std::endl;
            const std::vector<CNNLayerPtr> parents = getParentsRecursivelyExceptTypes(*child, {}, 1);
            for (const CNNLayerPtr& parent : parents) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              for (const CNNLayerPtr& parent : parents) {" << std::endl;
                if (parent->name == layer.name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                  if (parent->name == layer.name) {" << std::endl;
                    return 1;
                }
            }
            return -1;
        }

        const int result = onWeightsInDepth(*child);
        if (result != 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (result != 0) {" << std::endl;
            return result;
        }
    }
    return 0;
}

bool CNNNetworkHelper::onWeights(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  bool CNNNetworkHelper::onWeights(const CNNLayer& layer) {" << std::endl;
    const int result = onWeightsInDepth(layer);
    return result == 1;
}

size_t CNNNetworkHelper::getIndex(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  size_t CNNNetworkHelper::getIndex(const CNNLayer& layer) {" << std::endl;
    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    if (children.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (children.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "not supported";
    }

    for (size_t i = 0; i < children[0]->insData.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (size_t i = 0; i < children[0]->insData.size(); ++i) {" << std::endl;
        if (children[0]->insData[i].lock() != nullptr
                && children[0]->insData[i].lock()->getCreatorLayer().lock()->name == layer.name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                  && children[0]->insData[i].lock()->getCreatorLayer().lock()->name == layer.name) {" << std::endl;
            return i;
        }
    }

    THROW_IE_EXCEPTION << "not found";
}

std::vector<CNNLayerPtr> CNNNetworkHelper::transformFakeQuantizeToConst(TransformationContext& context,
                                                                        const CNNLayerPtr fakeQuantize,
                                                                        const Blob::Ptr weights,
                                                                        const std::string& constLayerName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                                                          const std::string& constLayerName) {" << std::endl;
    std::vector<CNNLayerPtr> constLayersToRemove;
    constLayersToRemove.reserve(fakeQuantize->insData.size());

    for (const DataWeakPtr& insDataWeak : fakeQuantize->insData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (const DataWeakPtr& insDataWeak : fakeQuantize->insData) {" << std::endl;
        const DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (insData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input data for FakeQuantize '" << fakeQuantize->name << "' is nullable";
        }
        const CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if (parent == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parent == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input layer for FakeQuantize '" << fakeQuantize->name << "' is nullable";
        }
        if (!CaselessEq<std::string>()(parent->type, "Const") || (parent->insData.size() != 0lu)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (!CaselessEq<std::string>()(parent->type, 'Const') || (parent->insData.size() != 0lu)) {" << std::endl;
            THROW_IE_EXCEPTION << "unexpected FakeQuantize input layer type " << parent->type << " for layer '"
                               << fakeQuantize->name << "' is nullable";
        }

        constLayersToRemove.push_back(parent);
    }

    for (const CNNLayerPtr& parent : constLayersToRemove) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (const CNNLayerPtr& parent : constLayersToRemove) {" << std::endl;
        CNNNetworkHelper::removeLayer(context.network, parent);
        context.removeLayer(*parent);
    }

    if (fakeQuantize->outData.size() != 1lu) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (fakeQuantize->outData.size() != 1lu) {" << std::endl;
        THROW_IE_EXCEPTION << "FakeQuantize " << fakeQuantize->name << " has several outputs";
    }

    const DataPtr outData = fakeQuantize->outData[0];
    if (outData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (outData == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "FakeQuantize output data is nullable";
    }

    // const Precision precision = outData->getPrecision();
    const auto inputTo = outData->getInputTo();
    std::vector<CNNLayerPtr> constLayers;
    for (auto it : inputTo) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (auto it : inputTo) {" << std::endl;
        const CNNLayerPtr child = it.second;
        if (child == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (child == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "child layer for FakeQuantize " << fakeQuantize->name << " is nullable";
        }

        constLayers.push_back(
            CNNNetworkHelper::addConstBetween(context.network, fakeQuantize, child, weights, constLayerName));
    }

    CNNNetworkHelper::removeLayer(context.network, fakeQuantize);
    context.removeLayer(*fakeQuantize);

    return constLayers;
}

void CNNNetworkHelper::setOutDataPrecision(const CNNLayer& layer, const Precision& precision) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::setOutDataPrecision(const CNNLayer& layer, const Precision& precision) {" << std::endl;
    for (const DataPtr& data : layer.outData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (const DataPtr& data : layer.outData) {" << std::endl;
        data->setPrecision(precision);
    }
}

void CNNNetworkHelper::setOutDataPrecision(const std::vector<CNNLayerPtr>& layers, const Precision& precision) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::setOutDataPrecision(const std::vector<CNNLayerPtr>& layers, const Precision& precision) {" << std::endl;
    for (const CNNLayerPtr layer : layers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (const CNNLayerPtr layer : layers) {" << std::endl;
        setOutDataPrecision(*layer, precision);
    }
}

void CNNNetworkHelper::setOutDataPrecision(const CNNLayer& beginLayer, const size_t branchWithEndBeforeLayer,
                                           const CNNLayer& endBeforeLayer, const Precision& precision) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                             const CNNLayer& endBeforeLayer, const Precision& precision) {" << std::endl;
    CNNLayerPtr child = std::make_shared<CNNLayer>(beginLayer);
    while (child->name != endBeforeLayer.name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      while (child->name != endBeforeLayer.name) {" << std::endl;
        CNNNetworkHelper::setOutDataPrecision(*child, precision);
        std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*child);
        if (child->name == beginLayer.name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (child->name == beginLayer.name) {" << std::endl;
            if (branchWithEndBeforeLayer >= children.size()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (branchWithEndBeforeLayer >= children.size()) {" << std::endl;
                THROW_IE_EXCEPTION << "branch with end before layer is out of children count " << children.size();
            }
            child = children[branchWithEndBeforeLayer];
        } else {
            if (children.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (children.size() != 1) {" << std::endl;
                THROW_IE_EXCEPTION << "not supported";
            }

            child = children[0];
        }
    }
}

bool CNNNetworkHelper::IsChild(const std::vector<CNNLayerPtr>& children,
                               const std::unordered_set<std::string>& layerTypes,
                               const std::unordered_set<std::string>& ignoreLayerTypes) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                 const std::unordered_set<std::string>& ignoreLayerTypes) {" << std::endl;
    for (const CNNLayerPtr child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (const CNNLayerPtr child : children) {" << std::endl;
        if (layerTypes.find(child->type) != layerTypes.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (layerTypes.find(child->type) != layerTypes.end()) {" << std::endl;
            return true;
        }
        if (ignoreLayerTypes.find(child->type) != ignoreLayerTypes.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (ignoreLayerTypes.find(child->type) != ignoreLayerTypes.end()) {" << std::endl;
            if (child->outData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (child->outData.size() != 1) {" << std::endl;
                return true;
            }
            if (IsChild(CNNNetworkHelper::getChildren(*child), layerTypes, ignoreLayerTypes)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (IsChild(CNNNetworkHelper::getChildren(*child), layerTypes, ignoreLayerTypes)) {" << std::endl;
                return true;
            }
        }
    }
    return false;
}

size_t CNNNetworkHelper::getOutputChannelsCount(const CNNLayer& layer, bool isOnWeights) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  size_t CNNNetworkHelper::getOutputChannelsCount(const CNNLayer& layer, bool isOnWeights) {" << std::endl;
    if (layer.outData.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer.outData.empty()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer " << layer.name << " doesn't have output tensors";
    }

    auto& data = layer.outData[0];
    if (isOnWeights) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (isOnWeights) {" << std::endl;
        if (data->getDims().empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (data->getDims().empty()) {" << std::endl;
            THROW_IE_EXCEPTION << "Invalid dimensions count (0) in output of " << layer.name << " layer on weights";
        }
        return data->getDims()[0];
    } else {
        if (data->getDims().empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (data->getDims().empty()) {" << std::endl;
            THROW_IE_EXCEPTION << "Invalid dimensions count (0) in output of " << layer.name << " layer on activations";
        }
        if (data->getDims().size() == 1ul) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (data->getDims().size() == 1ul) {" << std::endl;
            return data->getDims()[0];
        }
        return data->getDims()[1];
    }
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getLayers(const CNNLayer& parent, const CNNLayer& child) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  std::vector<CNNLayerPtr> CNNNetworkHelper::getLayers(const CNNLayer& parent, const CNNLayer& child) {" << std::endl;
    std::vector<CNNLayerPtr> layers;
    CNNLayerPtr tmpChild = std::make_shared<CNNLayer>(child);
    while (tmpChild != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      while (tmpChild != nullptr) {" << std::endl;
        const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*tmpChild);
        for (const CNNLayerPtr tmpParent : parents) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (const CNNLayerPtr tmpParent : parents) {" << std::endl;
            if (tmpParent->name == parent.name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (tmpParent->name == parent.name) {" << std::endl;
                return layers;
            }
        }

        if (parents.size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parents.size() == 0) {" << std::endl;
            THROW_IE_EXCEPTION << "not found";
        }

        if (parents.size() != 1ul) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parents.size() != 1ul) {" << std::endl;
            THROW_IE_EXCEPTION << "not supported";
        }

        layers.push_back(parents[0]);
        tmpChild = parents[0];
    }
    return layers;
}

Blob::Ptr CNNNetworkHelper::getBlob(CNNLayer* layer, const std::string& blobName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  Blob::Ptr CNNNetworkHelper::getBlob(CNNLayer* layer, const std::string& blobName) {" << std::endl;
    if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "layer is nullable";
    }
    if (layer->blobs.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->blobs.empty()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer '" << layer->name << "' does not have any blob";
    }
    if (blobName.empty() && (layer->blobs.size() != 1)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (blobName.empty() && (layer->blobs.size() != 1)) {" << std::endl;
        THROW_IE_EXCEPTION << "several blobs";
    }
    Blob::Ptr blob = blobName.empty() ? layer->blobs.begin()->second : layer->blobs[blobName];
    return blob;
}

Blob::Ptr CNNNetworkHelper::getBlob(CNNLayerPtr layer, const std::string& blobName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  Blob::Ptr CNNNetworkHelper::getBlob(CNNLayerPtr layer, const std::string& blobName) {" << std::endl;
    return getBlob(layer.get(), blobName);
}

std::shared_ptr<float> CNNNetworkHelper::getFloatData(const Blob::Ptr& srcBlob) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  std::shared_ptr<float> CNNNetworkHelper::getFloatData(const Blob::Ptr& srcBlob) {" << std::endl;
    if (srcBlob == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (srcBlob == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "Invalid blob";
    }

    const auto& precision = srcBlob->getTensorDesc().getPrecision();
    if (!isBlobPrecisionSupported(precision)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (!isBlobPrecisionSupported(precision)) {" << std::endl;
        THROW_IE_EXCEPTION << "precision '" << precision << "' is not supported";
    }

    const size_t dataSize = srcBlob->size();
    std::shared_ptr<float> floatPtr(new float[dataSize], std::default_delete<float[]>());

    if (precision == Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (precision == Precision::FP32) {" << std::endl;
        const float* srcData = srcBlob->buffer().as<float*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::FP16) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::FP16) {" << std::endl;
        const short* srcData = srcBlob->buffer().as<short*>();
        PrecisionUtils::f16tof32Arrays(floatPtr.get(), srcData, dataSize, 1.f, 0.f);
    } else if (precision == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::I8) {" << std::endl;
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::I8>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::U8) {" << std::endl;
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::I32) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::I32) {" << std::endl;
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::I64) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::I64) {" << std::endl;
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::I64>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }

    return floatPtr;
}

bool CNNNetworkHelper::isBlobPrecisionSupported(const Precision precision) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  bool CNNNetworkHelper::isBlobPrecisionSupported(const Precision precision) {" << std::endl;
    return (precision == Precision::FP32) ||
        (precision == Precision::FP16) ||
        (precision == Precision::I8) ||
        (precision == Precision::U8) ||
        (precision == Precision::I32) ||
        (precision == Precision::I64);
}

std::shared_ptr<float> CNNNetworkHelper::getFloatData(const CNNLayerPtr& layer, const std::string& blobName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  std::shared_ptr<float> CNNNetworkHelper::getFloatData(const CNNLayerPtr& layer, const std::string& blobName) {" << std::endl;
    const Blob::Ptr blob = getBlob(layer, blobName);
    if (blob == nullptr) THROW_IE_EXCEPTION << "Could not find blob '" << blobName << "' for layer " << layer->name;

    return getFloatData(blob);
}

void CNNNetworkHelper::fillBlobByFP32(Blob::Ptr& dstBlob, const float* srcData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::fillBlobByFP32(Blob::Ptr& dstBlob, const float* srcData) {" << std::endl;
    if (dstBlob == nullptr) THROW_IE_EXCEPTION << "Invalid blob";

    const auto& precision = dstBlob->getTensorDesc().getPrecision();
    const size_t dataSize = dstBlob->size();

    if (precision == Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (precision == Precision::FP32) {" << std::endl;
        float* dstData = dstBlob->buffer().as<float*>();
        std::copy(srcData, srcData + dataSize, dstData);
    } else if (precision == Precision::FP16) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::FP16) {" << std::endl;
        short* dstData = dstBlob->buffer().as<short*>();
        PrecisionUtils::f32tof16Arrays(dstData, srcData, dataSize, 1.f, 0.f);
    } else if (precision == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::I8) {" << std::endl;
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I8>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t i = 0ul; i < dataSize; ++i) {" << std::endl;
            dstData[i] = static_cast<PrecisionTrait<Precision::I8>::value_type>(std::roundf(srcData[i]));
        }
    } else if (precision == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::U8) {" << std::endl;
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t i = 0ul; i < dataSize; ++i) {" << std::endl;
            dstData[i] = static_cast<PrecisionTrait<Precision::U8>::value_type>(std::roundf(srcData[i]));
        }
    } else if (precision == Precision::I32) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::I32) {" << std::endl;
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t i = 0ul; i < dataSize; ++i) {" << std::endl;
            dstData[i] = static_cast<PrecisionTrait<Precision::I32>::value_type>(std::roundf(srcData[i]));
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }
}

std::shared_ptr<float> CNNNetworkHelper::convertFloatData(const float* srcData, const size_t dataSize,
                                                          const Precision precision) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                                            const Precision precision) {" << std::endl;
    std::shared_ptr<float> dstData(new float[dataSize], std::default_delete<float[]>());

    if (precision == Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (precision == Precision::FP32) {" << std::endl;
        std::copy(srcData, srcData + dataSize, dstData.get());
    } else if (precision == Precision::FP16) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::FP16) {" << std::endl;
        for (size_t i = 0ul; i < dataSize; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t i = 0ul; i < dataSize; ++i) {" << std::endl;
            dstData.get()[i] = PrecisionUtils::f16tof32(PrecisionUtils::f16tof32(srcData[i]));
        }
    } else if (precision == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::I8) {" << std::endl;
        for (size_t i = 0ul; i < dataSize; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t i = 0ul; i < dataSize; ++i) {" << std::endl;
            dstData.get()[i] =
                static_cast<float>(static_cast<PrecisionTrait<Precision::I8>::value_type>(std::roundf(srcData[i])));
        }
    } else if (precision == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::U8) {" << std::endl;
        for (size_t i = 0ul; i < dataSize; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t i = 0ul; i < dataSize; ++i) {" << std::endl;
            dstData.get()[i] =
                static_cast<float>(static_cast<PrecisionTrait<Precision::U8>::value_type>(std::roundf(srcData[i])));
        }
    } else if (precision == Precision::I32) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::I32) {" << std::endl;
        for (size_t i = 0ul; i < dataSize; ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t i = 0ul; i < dataSize; ++i) {" << std::endl;
            dstData.get()[i] =
                static_cast<float>(static_cast<PrecisionTrait<Precision::I32>::value_type>(std::roundf(srcData[i])));
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }

    return dstData;
}

void CNNNetworkHelper::fillBlobByFP32(const CNNLayerPtr& layer, const std::string& blobName, const float* srcData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::fillBlobByFP32(const CNNLayerPtr& layer, const std::string& blobName, const float* srcData) {" << std::endl;
    Blob::Ptr blob = getBlob(layer, blobName);
    return fillBlobByFP32(blob, srcData);
}

void CNNNetworkHelper::fillBlobByFP32(Blob::Ptr& dstBlob, float value) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::fillBlobByFP32(Blob::Ptr& dstBlob, float value) {" << std::endl;
    const auto& precision = dstBlob->getTensorDesc().getPrecision();
    const size_t dataSize = dstBlob->size();

    if (precision == Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (precision == Precision::FP32) {" << std::endl;
        float* dstData = dstBlob->buffer().as<float*>();
        std::fill(dstData, dstData + dataSize, value);
    } else if (precision == Precision::FP16) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::FP16) {" << std::endl;
        short* dstData = dstBlob->buffer().as<short*>();
        const short s_value = PrecisionUtils::f32tof16(value);
        std::fill(dstData, dstData + dataSize, s_value);
    } else if (precision == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::I8) {" << std::endl;
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I8>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<PrecisionTrait<Precision::I8>::value_type>(value));
    } else if (precision == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::U8) {" << std::endl;
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<PrecisionTrait<Precision::U8>::value_type>(value));
    } else if (precision == Precision::I32) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (precision == Precision::I32) {" << std::endl;
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<PrecisionTrait<Precision::I32>::value_type>(value));
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }
}

CNNLayerPtr CNNNetworkHelper::getParent(const CNNLayer& layer, const size_t index, const std::string& ignoreLayerType) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  CNNLayerPtr CNNNetworkHelper::getParent(const CNNLayer& layer, const size_t index, const std::string& ignoreLayerType) {" << std::endl;
    if (index >= layer.insData.size()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (index >= layer.insData.size()) {" << std::endl;
        return nullptr;
    }

    DataPtr inputLayerData = layer.insData[index].lock();
    if (inputLayerData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (inputLayerData == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "input data is absent";
    }

    CNNLayerPtr inputLayer;
    do {
        inputLayer = inputLayerData->getCreatorLayer().lock();
        if (!inputLayer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (!inputLayer) {" << std::endl;
            THROW_IE_EXCEPTION << "input is absent";
        }

        if (inputLayer->type != ignoreLayerType) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (inputLayer->type != ignoreLayerType) {" << std::endl;
            break;
        }

        if (inputLayer->insData.size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (inputLayer->insData.size() == 0) {" << std::endl;
            inputLayer = nullptr;
            break;
        }

        if (inputLayer->insData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (inputLayer->insData.size() != 1) {" << std::endl;
            THROW_IE_EXCEPTION << "too much branches";
        }

        inputLayerData = inputLayer->insData[0].lock();
        if (inputLayerData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (inputLayerData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input data is absent";
        }
    } while (true);

    return inputLayer;
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getParents(const CNNLayer& layer, const std::string& exceptionLayerName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  std::vector<CNNLayerPtr> CNNNetworkHelper::getParents(const CNNLayer& layer, const std::string& exceptionLayerName) {" << std::endl;
    std::vector<CNNLayerPtr> parents;
    for (const DataWeakPtr insDataWeak : layer.insData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (const DataWeakPtr insDataWeak : layer.insData) {" << std::endl;
        const DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (insData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input data is absent";
        }

        CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if (parent == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parent == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input layer is absent";
        }

        if (exceptionLayerName.empty() || parent->name != exceptionLayerName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (exceptionLayerName.empty() || parent->name != exceptionLayerName) {" << std::endl;
            parents.push_back(parent);
        }
    }
    return parents;
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getParentsRecursivelyExceptTypes(
    const CNNLayer& layer, const std::unordered_set<std::string>& exceptionLayerTypes, const int portIndex) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      const CNNLayer& layer, const std::unordered_set<std::string>& exceptionLayerTypes, const int portIndex) {" << std::endl;
    std::vector<CNNLayerPtr> parents;
    size_t i = 0ul;
    for (DataWeakPtr insDataWeak : layer.insData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (DataWeakPtr insDataWeak : layer.insData) {" << std::endl;
        if (insDataWeak.expired()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (insDataWeak.expired()) {" << std::endl;
            continue;
        }

        const DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (insData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "input data is absent";
        }

        CNNLayerWeakPtr parentWeak = insData->getCreatorLayer();
        if (parentWeak.expired()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parentWeak.expired()) {" << std::endl;
            continue;
        }

        if ((portIndex == -1) || (portIndex == i)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if ((portIndex == -1) || (portIndex == i)) {" << std::endl;
            CNNLayerPtr parent = parentWeak.lock();
            if (parent == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (parent == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "input layer is absent";
            }

            if (exceptionLayerTypes.find(parent->type) != exceptionLayerTypes.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (exceptionLayerTypes.find(parent->type) != exceptionLayerTypes.end()) {" << std::endl;
                const std::vector<CNNLayerPtr> tmpParents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(*parent, exceptionLayerTypes);
                parents.insert(parents.end(), tmpParents.begin(), tmpParents.end());
            } else {
                parents.push_back(parent);
            }
        }

        i++;
    }
    return parents;
}

size_t CNNNetworkHelper::getInputChannelsCount(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  size_t CNNNetworkHelper::getInputChannelsCount(const CNNLayer& layer) {" << std::endl;
    if (layer.insData.size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer.insData.size() == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "There are no input layers";
    }

    const DataPtr insertData = layer.insData[0].lock();
    if (insertData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (insertData == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "insert data is absent";
    }

    switch (insertData->getLayout()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      switch (insertData->getLayout()) {" << std::endl;
    case Layout::NC:
    case Layout::NCHW:
    case Layout::NCDHW: {
        return insertData->getDims()[1];
    }
    case Layout::CHW: {
        if (insertData->getDims().size() != 3lu) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (insertData->getDims().size() != 3lu) {" << std::endl;
            THROW_IE_EXCEPTION << "Unexpected dimensions size " << insertData->getDims().size() << " for layer "
                               << layer.name;
        }

        // Actually MO assumes NCH layout for 3D blobs, so we get channels count from dimension 1
        return insertData->getDims()[1];
    }
    default: {
        THROW_IE_EXCEPTION << "Not supported layout " << insertData->getLayout();
    }
    }
}

size_t CNNNetworkHelper::getParamOutput(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  size_t CNNNetworkHelper::getParamOutput(const CNNLayer& layer) {" << std::endl;
    if (!layer.CheckParamPresence("output")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (!layer.CheckParamPresence('output')) {" << std::endl;
        THROW_IE_EXCEPTION << "convolution parameter 'output' is absent";
    }
    return layer.GetParamAsUInt("output");
}

size_t CNNNetworkHelper::getKernelSize(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  size_t CNNNetworkHelper::getKernelSize(const CNNLayer& layer) {" << std::endl;
    if (!layer.CheckParamPresence("kernel")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (!layer.CheckParamPresence('kernel')) {" << std::endl;
        THROW_IE_EXCEPTION << "convolution parameter 'kernel' is absent";
    }
    const auto dims = layer.GetParamAsUInts("kernel");
    if (dims.size() == 2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (dims.size() == 2) {" << std::endl;
        return dims[0] * dims[1];
    } else if (dims.size() == 3) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      } else if (dims.size() == 3) {" << std::endl;
        return dims[0] * dims[1] * dims[2];
    } else {
        THROW_IE_EXCEPTION << "kernel dimensions are not correct";
    }
}

void CNNNetworkHelper::renameLayer(ICNNNetwork& net, const std::string& currentName, const std::string& newName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::renameLayer(ICNNNetwork& net, const std::string& currentName, const std::string& newName) {" << std::endl;
    CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
    if (netImpl == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (netImpl == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected network type";
    }

    netImpl->renameLayer(currentName, newName);
}

CNNLayerPtr CNNNetworkHelper::addLayer(
        TransformationContext& context,
        const CNNLayerPtr parent,
        const CNNLayerPtr child,
        const CNNLayerPtr newLayer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          const CNNLayerPtr newLayer) {" << std::endl;
    DataPtr outData;
    Precision precision;
    if (parent != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (parent != nullptr) {" << std::endl;
        // Searching the connection between the layers
        int l1_out_i = 0;
        if (child != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (child != nullptr) {" << std::endl;
            for (; l1_out_i < parent->outData.size(); l1_out_i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              for (; l1_out_i < parent->outData.size(); l1_out_i++) {" << std::endl;
                if (parent->outData[l1_out_i]->getInputTo().find(child->name) !=
                    parent->outData[l1_out_i]->getInputTo().end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                      parent->outData[l1_out_i]->getInputTo().end()) {" << std::endl;
                    break;
                }
            }
        }
        if (l1_out_i == parent->outData.size()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (l1_out_i == parent->outData.size()) {" << std::endl;
            if (child != nullptr)
                THROW_IE_EXCEPTION << "Can't find layer " << child->name << " among layer " << parent->name << " outputs";
            else
                THROW_IE_EXCEPTION << "Layer '" << parent->name << "' has invalid output";
        }

        outData = parent->outData[l1_out_i];
        precision = context.getOriginalLayerPrecision(parent->name, outData->getName());
        if (precision == Precision::UNSPECIFIED) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (precision == Precision::UNSPECIFIED) {" << std::endl;
            if (child != nullptr)
                precision = child->precision;
            else if (context.network.getPrecision() != Precision::MIXED)
                precision = context.network.getPrecision();
            else
                precision = Precision::FP32;
        }
    } else {
        // TODO: FIXME
        precision = Precision::FP32;
        outData = nullptr;
    }
    addLayerToCNNNetworkAfterData(outData, newLayer, child != nullptr ? child->name : "", context.network);

    CNNNetworkHelper::setOutDataPrecision(*newLayer, precision);
    return newLayer;
}

void CNNNetworkHelper::replaceLayer(TransformationContext& context, const CNNLayerPtr source, const CNNLayerPtr target) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::replaceLayer(TransformationContext& context, const CNNLayerPtr source, const CNNLayerPtr target) {" << std::endl;
    CNNNetworkImpl* networkImpl = dynamic_cast<CNNNetworkImpl*>(&context.network);
    networkImpl->removeLayer(source->name);

    std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*source);
    for (CNNLayerPtr parent : parents) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (CNNLayerPtr parent : parents) {" << std::endl;
        for (size_t outDataIndex = 0ul; outDataIndex < parent->outData.size(); ++outDataIndex) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t outDataIndex = 0ul; outDataIndex < parent->outData.size(); ++outDataIndex) {" << std::endl;
            const DataPtr outData = parent->outData[outDataIndex];
            std::map<std::string, CNNLayerPtr>& inputTo = outData->getInputTo();
            inputTo[source->name] = target;
            target->insData.push_back(outData);
        }
    }

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*source);

    target->outData.resize(source->outData.size());
    for (size_t outDataIndex = 0ul; outDataIndex < source->outData.size(); ++outDataIndex) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (size_t outDataIndex = 0ul; outDataIndex < source->outData.size(); ++outDataIndex) {" << std::endl;
        const DataPtr outData = source->outData[outDataIndex];
        networkImpl->removeData(outData->getName());

        DataPtr newOutData(new Data(outData->getName(), outData->getTensorDesc()));
        newOutData->getCreatorLayer() = target;
        target->outData[outDataIndex] = newOutData;
        networkImpl->addData(newOutData->getName().c_str(), newOutData);

        std::map<std::string, CNNLayerPtr> inputTo = outData->getInputTo();
        for (const auto it : inputTo) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (const auto it : inputTo) {" << std::endl;
            const CNNLayerPtr child = it.second;
            newOutData->getInputTo().emplace(it.first, child);

            {
                for (const CNNLayerPtr child : children) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                  for (const CNNLayerPtr child : children) {" << std::endl;
                    for (size_t insDataIndex = 0ul; insDataIndex < child->insData.size(); ++insDataIndex) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                      for (size_t insDataIndex = 0ul; insDataIndex < child->insData.size(); ++insDataIndex) {" << std::endl;
                        DataPtr insData = child->insData[insDataIndex].lock();
                        if (insData->getCreatorLayer().lock()->name == source->name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                          if (insData->getCreatorLayer().lock()->name == source->name) {" << std::endl;
                            const auto it = target->outData[outDataIndex];
                            child->insData[insDataIndex] = newOutData;
                        }
                    }
                }
            }
        }
        outData->getInputTo().clear();
    }

    context.network.addLayer(target);
}

CNNLayerPtr CNNNetworkHelper::addScaleShiftBetween(TransformationContext& context, const CNNLayerPtr parent,
                                                   const CNNLayerPtr child,
                                                   const DequantizationDetails& dequantizationDetails,
                                                   const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                                     const std::string& name) {" << std::endl;
    if (parent == nullptr)
        THROW_IE_EXCEPTION << "Parent layer is nullable";

    if (child && (child->type == "ScaleShift") && (CNNNetworkHelper::getParents(*child).size() == 1)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (child && (child->type == 'ScaleShift') && (CNNNetworkHelper::getParents(*child).size() == 1)) {" << std::endl;
        auto scalesIt = child->blobs.find("weights");
        if (scalesIt == child->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (scalesIt == child->blobs.end()) {" << std::endl;
            THROW_IE_EXCEPTION << "weights for layer " << child->name << " was not found";
        }
        const std::shared_ptr<float> scales = CNNNetworkHelper::getFloatData(scalesIt->second);
        std::vector<float> updatedScales(scalesIt->second->size());
        for (size_t i = 0ul; i < updatedScales.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t i = 0ul; i < updatedScales.size(); ++i) {" << std::endl;
            updatedScales[i] = scales.get()[i] * dequantizationDetails.scales[i];
        }
        CNNNetworkHelper::updateBlobs(*child, "weights", updatedScales);

        auto shiftsIt = child->blobs.find("biases");
        if (shiftsIt != child->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (shiftsIt != child->blobs.end()) {" << std::endl;
            const std::shared_ptr<float> shifts = CNNNetworkHelper::getFloatData(shiftsIt->second);
            std::vector<float> updatedShifts(shiftsIt->second->size());
            for (size_t i = 0ul; i < updatedShifts.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              for (size_t i = 0ul; i < updatedShifts.size(); ++i) {" << std::endl;
                updatedShifts[i] = scales.get()[i] * dequantizationDetails.shifts[i] + shifts.get()[i];
            }
            CNNNetworkHelper::updateBlobs(*child, "biases", updatedShifts);
        }

        return child;
    }

    // Searching the connection between the layers
    int l1_out_i = 0;
    if (child != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (child != nullptr) {" << std::endl;
        for (; l1_out_i < parent->outData.size(); l1_out_i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (; l1_out_i < parent->outData.size(); l1_out_i++) {" << std::endl;
            if (parent->outData[l1_out_i]->getInputTo().find(child->name) !=
                parent->outData[l1_out_i]->getInputTo().end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                  parent->outData[l1_out_i]->getInputTo().end()) {" << std::endl;
                break;
            }
        }
    }
    if (l1_out_i == parent->outData.size()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (l1_out_i == parent->outData.size()) {" << std::endl;
        if (child != nullptr)
            THROW_IE_EXCEPTION << "Can't find layer " << child->name << " among layer " << parent->name << " outputs";
        else
            THROW_IE_EXCEPTION << "Layer '" << parent->name << "' has invalid output";
    }

    DataPtr outData = parent->outData[l1_out_i];

    std::string layerName = name.empty() ? (child != nullptr ? (parent->name + "_ScaleShift_" + child->name)
                                                             : (parent->name + "_ScaleShift"))
                                         : name;

    Precision ssPrecision = context.getOriginalLayerPrecision(parent->name, outData->getName());
    if (ssPrecision == Precision::UNSPECIFIED) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (ssPrecision == Precision::UNSPECIFIED) {" << std::endl;
        if (child != nullptr)
            ssPrecision = child->precision;
        else if (context.network.getPrecision() != Precision::MIXED)
            ssPrecision = context.network.getPrecision();
        else
            ssPrecision = Precision::FP32;
    }

    LayerParams ssCnnLayerParams {layerName, "ScaleShift", ssPrecision};
    CNNLayerPtr ssCnnLayer(new ScaleShiftLayer(ssCnnLayerParams));

    const std::vector<size_t> dims = outData->getDims();
    if ((dims.size() > 1) && (dims[1] != dequantizationDetails.channelsCount)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if ((dims.size() > 1) && (dims[1] != dequantizationDetails.channelsCount)) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected parent channels count " << dims[1];
    }
    addLayerToCNNNetworkAfterData(outData, ssCnnLayer, child != nullptr ? child->name : "", context.network);

    {
        ScaleShiftLayer* scshLayer = dynamic_cast<ScaleShiftLayer*>(ssCnnLayer.get());
        if (scshLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (scshLayer == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << ssCnnLayer->name << " is not instance of ScaleShiftLayer class";
        }
        fillInScaleShift(scshLayer, dequantizationDetails.channelsCount, dequantizationDetails.scales.data(),
                         dequantizationDetails.shifts.data());
    }

    CNNNetworkHelper::setOutDataPrecision(*ssCnnLayer, ssPrecision);
    return ssCnnLayer;
}

CNNLayerPtr CNNNetworkHelper::addConstBetween(ICNNNetwork& net, const CNNLayerPtr layer1, const CNNLayerPtr layer2,
                                              const Blob::Ptr customBlob, const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                                const Blob::Ptr customBlob, const std::string& name) {" << std::endl;
    if (layer1 == nullptr)
        THROW_IE_EXCEPTION << "First layer is nullable";
    // Searching the connection between the layers
    int l1_out_i = 0;
    if (layer2 != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer2 != nullptr) {" << std::endl;
        for (; l1_out_i < layer1->outData.size(); l1_out_i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (; l1_out_i < layer1->outData.size(); l1_out_i++) {" << std::endl;
            if (layer1->outData[l1_out_i]->getInputTo().find(layer2->name) !=
                layer1->outData[l1_out_i]->getInputTo().end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                  layer1->outData[l1_out_i]->getInputTo().end()) {" << std::endl;
                break;
            }
        }
    }

    if (l1_out_i == layer1->outData.size()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (l1_out_i == layer1->outData.size()) {" << std::endl;
        if (layer2 != nullptr)
            THROW_IE_EXCEPTION << "Can't find layer " << layer2->name << " among layer " << layer1->name << " outputs";
        else
            THROW_IE_EXCEPTION << "Layer " << layer1->name << " has invalid outputs";
    }

    DataPtr outData = layer1->outData[l1_out_i];

    std::string layerName = name.empty() ? layer1->name + "_Const" : name;
    CNNLayerPtr layer(new CNNLayer({layerName, "Const", customBlob->getTensorDesc().getPrecision()}));

    addLayerToCNNNetworkAfterData(outData, layer, layer2 != nullptr ? layer2->name : "", net);
    layer->blobs.emplace("custom", customBlob);
    layer->outData[0]->setPrecision(customBlob->getTensorDesc().getPrecision());
    return layer;
}

void CNNNetworkHelper::addLayerToCNNNetworkAfterData(
    DataPtr parentOutData,
    CNNLayer::Ptr layer,
    const std::string& nextLayerName,
    ICNNNetwork& net) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      ICNNNetwork& net) {" << std::endl;
    CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
    if (netImpl == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (netImpl == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected network type";
    }

    CNNLayerPtr nextLayer;
    if (!nextLayerName.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (!nextLayerName.empty()) {" << std::endl;
        netImpl->getLayerByName(nextLayerName.c_str(), nextLayer, nullptr);
    }

    if (layer && (nextLayerName.empty() || (parentOutData == nullptr) ||
                  (parentOutData->getInputTo().find(nextLayerName) != parentOutData->getInputTo().end()))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                    (parentOutData->getInputTo().find(nextLayerName) != parentOutData->getInputTo().end()))) {" << std::endl;
        const TensorDesc& parentTensorDesc =
            parentOutData != nullptr ? parentOutData->getTensorDesc() : nextLayer->insData[0].lock()->getTensorDesc();
        DataPtr newEdgeAfterLayer(new Data(layer->name, parentTensorDesc));
        newEdgeAfterLayer->setName(layer->name);
        newEdgeAfterLayer->getCreatorLayer() = layer;
        newEdgeAfterLayer->getInputTo().clear();

        CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
        if (netImpl == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (netImpl == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "unexpected network type";
        }
        netImpl->addData(layer->name.c_str(), newEdgeAfterLayer);
        netImpl->addLayer(layer);

        if (parentOutData != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parentOutData != nullptr) {" << std::endl;
            parentOutData->getInputTo()[layer->name] = layer;
            layer->insData.push_back(parentOutData);
        }
        layer->outData.push_back(newEdgeAfterLayer);

        if (!nextLayerName.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (!nextLayerName.empty()) {" << std::endl;
            // CNNLayerPtr nextLayer = parentOutData->getInputTo()[nextLayerName];
            newEdgeAfterLayer->getInputTo()[nextLayerName] = nextLayer;
            if (parentOutData != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (parentOutData != nullptr) {" << std::endl;
                parentOutData->getInputTo().erase(nextLayerName);
                for (size_t i = 0; i < nextLayer->insData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                  for (size_t i = 0; i < nextLayer->insData.size(); i++) {" << std::endl;
                    if (nextLayer->insData[i].lock() == parentOutData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                      if (nextLayer->insData[i].lock() == parentOutData) {" << std::endl;
                        nextLayer->insData[i] = newEdgeAfterLayer;
                    }
                }
            } else {
                // TODO: why new?
                nextLayer->insData.push_back(newEdgeAfterLayer);
            }
        } else {
            netImpl->removeOutput(parentOutData->getCreatorLayer().lock()->name);
            netImpl->addData(layer->name.c_str(), newEdgeAfterLayer);
            netImpl->addOutput(layer->name);
        }
    } else {
        THROW_IE_EXCEPTION << "Invalid argument";
    }
}

void CNNNetworkHelper::fillInScaleShift(ScaleShiftLayer* layer, const size_t channels, const float* scales,
                                        const float* shifts) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                          const float* shifts) {" << std::endl;
    if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "ScaleShiftLayer is nullable";
    }

    layer->_weights = makeNewBlobPtr({layer->precision, {channels}, Layout::C});
    layer->_weights->allocate();
    fillBlobByFP32(layer->_weights, scales);
    layer->blobs["weights"] = layer->_weights;

    layer->_biases = makeNewBlobPtr({layer->precision, {channels}, Layout::C});
    layer->_biases->allocate();
    fillBlobByFP32(layer->_biases, shifts);
    layer->blobs["biases"] = layer->_biases;
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getChildren(const CNNLayer& layer, const std::string& exceptionLayerName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  std::vector<CNNLayerPtr> CNNNetworkHelper::getChildren(const CNNLayer& layer, const std::string& exceptionLayerName) {" << std::endl;
    std::vector<CNNLayerPtr> children;
    for (const DataPtr outData : layer.outData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (const DataPtr outData : layer.outData) {" << std::endl;
        const std::map<std::string, CNNLayerPtr>& inputTo = outData->getInputTo();
        for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {" << std::endl;
            CNNLayerPtr child = it->second;
            if (exceptionLayerName.empty() || child->name != exceptionLayerName) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (exceptionLayerName.empty() || child->name != exceptionLayerName) {" << std::endl;
                children.push_back(child);
            }
        }
    }
    return children;
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getChildrenRecursivelyExceptTypes(
    const CNNLayer& layer, const std::unordered_set<std::string>& exceptionLayerTypes) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      const CNNLayer& layer, const std::unordered_set<std::string>& exceptionLayerTypes) {" << std::endl;
    std::vector<CNNLayerPtr> children;
    for (const DataPtr outData : layer.outData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (const DataPtr outData : layer.outData) {" << std::endl;
        const std::map<std::string, CNNLayerPtr>& inputTo = outData->getInputTo();
        for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {" << std::endl;
            CNNLayerPtr child = it->second;
            if (exceptionLayerTypes.find(child->type) != exceptionLayerTypes.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (exceptionLayerTypes.find(child->type) != exceptionLayerTypes.end()) {" << std::endl;
                const std::vector<CNNLayerPtr> tmpChildren =
                    getChildrenRecursivelyExceptTypes(*child, exceptionLayerTypes);
                children.insert(children.end(), tmpChildren.begin(), tmpChildren.end());
                continue;
            }

            children.push_back(child);
        }
    }
    return children;
}

void CNNNetworkHelper::checkConstWithBlobs(const CNNLayerPtr layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::checkConstWithBlobs(const CNNLayerPtr layer) {" << std::endl;
    if (layer->type != "Const") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->type != 'Const') {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected layer type '" << layer->name << "'";
    }
    if (layer->blobs.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->blobs.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected blobs count " << layer->blobs.size() << " for layer '" << layer->name << "'";
    }
    if (layer->insData.size() != 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->insData.size() != 0) {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected inputs count " << layer->insData.size() << " for layer '" << layer->name
                           << "'";
    }
    if (layer->outData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->outData.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected outputs count " << layer->outData.size() << " for layer '" << layer->name
                           << "'";
    }
}

void CNNNetworkHelper::checkQuantizeOnWeights(const CNNLayerPtr layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::checkQuantizeOnWeights(const CNNLayerPtr layer) {" << std::endl;
    if (layer->type != "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->type != 'FakeQuantize') {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected layer type '" << layer->name << "'";
    }
    if (layer->blobs.size() != 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->blobs.size() != 0) {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected blobs count " << layer->blobs.size() << " for layer '" << layer->name << "'";
    }
    if (layer->insData.size() != 5) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->insData.size() != 5) {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected inputs count " << layer->insData.size() << " for layer '" << layer->name
                           << "'";
    }
    if (layer->outData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->outData.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected outputs count " << layer->outData.size() << " for layer '" << layer->name
                           << "'";
    }
}

void CNNNetworkHelper::updateInput(CNNNetworkImpl* network, CNNLayerPtr& layer, DataPtr outData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::updateInput(CNNNetworkImpl* network, CNNLayerPtr& layer, DataPtr outData) {" << std::endl;
    if (!CaselessEq<std::string>()(layer->type, "Input")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (!CaselessEq<std::string>()(layer->type, 'Input')) {" << std::endl;
        return;
    }

    InputInfo::Ptr inputInfo = network->getInput(layer->name);
    if (inputInfo->name() == layer->name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (inputInfo->name() == layer->name) {" << std::endl;
        inputInfo->setInputData(outData);
    }
}

size_t CNNNetworkHelper::disconnectLayers(CNNNetworkImpl* network, const CNNLayerPtr& parentLayer,
                                          const CNNLayerPtr& childLayer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                            const CNNLayerPtr& childLayer) {" << std::endl;
    bool wasFound = false;
    for (auto dataIt = parentLayer->outData.begin(); dataIt != parentLayer->outData.end(); ++dataIt) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (auto dataIt = parentLayer->outData.begin(); dataIt != parentLayer->outData.end(); ++dataIt) {" << std::endl;
        auto data = *dataIt;
        for (auto inputIt = data->getInputTo().begin(); inputIt != data->getInputTo().end(); ++inputIt) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (auto inputIt = data->getInputTo().begin(); inputIt != data->getInputTo().end(); ++inputIt) {" << std::endl;
            auto currentChildLayer = inputIt->second;
            if (currentChildLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (currentChildLayer == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "Output layer for '" << parentLayer->name << "'is absent";
            }
            if (currentChildLayer->name == childLayer->name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (currentChildLayer->name == childLayer->name) {" << std::endl;
                const DataPtr dataToRemove = network->getData(data->getName().c_str());
                if (!dataToRemove) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                  if (!dataToRemove) {" << std::endl;
                    THROW_IE_EXCEPTION << "there is not data to remove";
                }

                data->getInputTo().erase(inputIt);
                wasFound = true;
                break;
            }
        }

        if (wasFound) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (wasFound) {" << std::endl;
            break;
        }
    }
    if (!wasFound) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (!wasFound) {" << std::endl;
        THROW_IE_EXCEPTION << "Output layer '" << childLayer->name << "' was not found for '" << parentLayer->name
                           << "'";
    }

    wasFound = false;
    for (auto it = childLayer->insData.begin(); it != childLayer->insData.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (auto it = childLayer->insData.begin(); it != childLayer->insData.end(); ++it) {" << std::endl;
        auto data = it->lock();
        if (data == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (data == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Input layer data for '" << childLayer->name << "'is absent";
        }
        auto currentParentLayer = data->getCreatorLayer().lock();
        if (currentParentLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (currentParentLayer == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Input layer for '" << childLayer->name << "'is absent";
        }
        if (currentParentLayer->name == parentLayer->name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (currentParentLayer->name == parentLayer->name) {" << std::endl;
            childLayer->insData.erase(it);
            wasFound = true;
            break;
        }
    }
    if (!wasFound) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (!wasFound) {" << std::endl;
        THROW_IE_EXCEPTION << "Input layer '" << parentLayer->name << "' was not found for '" << childLayer->name
                           << "'";
    }
    return 0;
}

size_t CNNNetworkHelper::getInputIndex(const CNNLayerPtr& childLayer, const CNNLayerPtr& parentLayer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  size_t CNNNetworkHelper::getInputIndex(const CNNLayerPtr& childLayer, const CNNLayerPtr& parentLayer) {" << std::endl;
    for (size_t index = 0; index < childLayer->insData.size(); ++index) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (size_t index = 0; index < childLayer->insData.size(); ++index) {" << std::endl;
        DataPtr currentParenData = childLayer->insData[index].lock();
        if (currentParenData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (currentParenData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "parent layer data is absent";
        }
        CNNLayerPtr currentParrentLayer = currentParenData->getCreatorLayer().lock();
        if (currentParrentLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (currentParrentLayer == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "parent layer is absent";
        }
        if (currentParrentLayer->name == parentLayer->name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (currentParrentLayer->name == parentLayer->name) {" << std::endl;
            return index;
        }
    }

    THROW_IE_EXCEPTION << "parent layer was not found";
}

void CNNNetworkHelper::removeLayer(ICNNNetwork& network, const CNNLayerPtr& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  void CNNNetworkHelper::removeLayer(ICNNNetwork& network, const CNNLayerPtr& layer) {" << std::endl;
    details::CNNNetworkImpl* networkImpl = dynamic_cast<details::CNNNetworkImpl*>(&network);
    if (networkImpl == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (networkImpl == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected network type";
    }

    if (layer->outData.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->outData.size() > 1) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer '" << layer->name << "' has too many outputs " << layer->outData.size();
    }

    if (layer->insData.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->insData.size() > 1) {" << std::endl;
        do {
            DataPtr data = layer->insData[0].lock();
            if (data == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (data == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "Layer's inserted data is nullptr";
            }
            CNNLayerPtr parentLayer = data->getCreatorLayer().lock();
            if (parentLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (parentLayer == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "Layer's parent layer is nullptr";
            }
            CNNNetworkHelper::removeLayer(network, parentLayer);
        } while (!layer->insData.empty());
    }

    DataPtr childData;
    std::vector<CNNLayerPtr> children;
    std::vector<size_t> childrenIndexes;
    if (layer->outData.size() > 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->outData.size() > 0) {" << std::endl;
        childData = layer->outData[0];
        auto inputTo = childData->getInputTo();
        if (inputTo.size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (inputTo.size() == 0) {" << std::endl;
            std::vector<CNNLayerPtr> parents = getParents(*layer);
            if (parents.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (parents.size() != 1) {" << std::endl;
                THROW_IE_EXCEPTION << "not possible remove output layer with several parents";
            }
            networkImpl->addOutput(parents[0]->name);
            CNNNetworkImpl* networkImpl = dynamic_cast<CNNNetworkImpl*>(&network);
            networkImpl->removeOutput(layer->name);
        } else {
            for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {" << std::endl;
                children.push_back(it->second);
                childrenIndexes.push_back(getInputIndex(it->second, layer));
                disconnectLayers(networkImpl, layer, it->second);
            }
        }
    }

    if (layer->insData.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->insData.size() > 1) {" << std::endl;
        // TODO: implement
        THROW_IE_EXCEPTION << "not implemented";
    }

    DataPtr parentData;
    CNNLayerPtr parentLayer;
    if (layer->insData.size() > 0) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer->insData.size() > 0) {" << std::endl;
        // remove connections with parent layers
        parentData = layer->insData[0].lock();
        if (parentData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parentData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Input data is absent";
        }
        parentLayer = parentData->getCreatorLayer().lock();
        if (parentLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parentLayer == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Input layer for '" << layer->name << "' is absent";
        }

        const size_t ouputLayerOutDataIndex = disconnectLayers(networkImpl, parentLayer, layer);
        if (ouputLayerOutDataIndex >= parentLayer->outData.size()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (ouputLayerOutDataIndex >= parentLayer->outData.size()) {" << std::endl;
            THROW_IE_EXCEPTION << "Index " << ouputLayerOutDataIndex << " out of range output ports count "
                               << parentLayer->outData.size() << " for layer " << parentLayer->name;
        }

        for (size_t index = 0; index < children.size(); ++index) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (size_t index = 0; index < children.size(); ++index) {" << std::endl;
            CNNLayerPtr childLayer = children[index];
            const size_t childInputIndex = childrenIndexes[index];

            DataPtr outData = parentLayer->outData[ouputLayerOutDataIndex];
            outData->getInputTo().emplace(childLayer->name, childLayer);
            childLayer->insData.insert(childLayer->insData.begin() + childInputIndex, outData);

            updateInput(networkImpl, parentLayer, outData);
        }
    }

    networkImpl->removeData(layer->name);
    networkImpl->removeLayer(layer->name);
}

bool CNNNetworkHelper::isWeightsSupported(const CNNLayer& layer) noexcept {
    if (layer.insData.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer.insData.size() > 1) {" << std::endl;
        CNNLayerPtr weightsLayer = CNNNetworkHelper::getParent(layer, 1);
        if (weightsLayer == nullptr)
            return false;
        if ((weightsLayer->type == "Const") || (weightsLayer->type == "FakeQuantize")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if ((weightsLayer->type == 'Const') || (weightsLayer->type == 'FakeQuantize')) {" << std::endl;
            return true;
        }

        if (weightsLayer->type == "ScaleShift") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (weightsLayer->type == 'ScaleShift') {" << std::endl;
            const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*weightsLayer);
            if (parents.size() != 1ul) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (parents.size() != 1ul) {" << std::endl;
                return false;
            }

            return (parents[0]->type == "FakeQuantize") || (parents[0]->type == "Const");
        }

        return false;
    } else {
        return layer.blobs.find("weights") != layer.blobs.end();
    }
}

Blob::Ptr CNNNetworkHelper::getWeights(
        const CNNLayer& layer,
        const bool roundQuantizedValues,
        const std::vector<float>& weightsShiftPerChannel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          const std::vector<float>& weightsShiftPerChannel) {" << std::endl;
    if (layer.insData.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer.insData.size() > 1) {" << std::endl;
        CNNLayerPtr weightsLayer = CNNNetworkHelper::getParent(layer, 1);
        if (weightsLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (weightsLayer == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Convolution weights const layer are absent";
        }

        if (weightsLayer->type == "Const") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (weightsLayer->type == 'Const') {" << std::endl;
            CNNNetworkHelper::checkConstWithBlobs(weightsLayer);
            return weightsLayer->blobs.find("custom")->second;
        } else if (weightsLayer->type == "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          } else if (weightsLayer->type == 'FakeQuantize') {" << std::endl;
            return CNNNetworkHelper::quantizeWeights(*weightsLayer, roundQuantizedValues, Precision::UNSPECIFIED, weightsShiftPerChannel);
        } else if (weightsLayer->type == "ScaleShift") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          } else if (weightsLayer->type == 'ScaleShift') {" << std::endl;
            const CNNLayerPtr parent = CNNNetworkHelper::getParent(*weightsLayer);
            if (parent == nullptr)
                THROW_IE_EXCEPTION << "Layer '" << weightsLayer->name << "' does not have parent";
            if (parent->type == "FakeQuantize") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (parent->type == 'FakeQuantize') {" << std::endl;
                return CNNNetworkHelper::quantizeWeights(*parent, roundQuantizedValues, Precision::UNSPECIFIED, weightsShiftPerChannel);
            } else if (parent->type == "Const") {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              } else if (parent->type == 'Const') {" << std::endl;
                CNNNetworkHelper::checkConstWithBlobs(parent);
                return CNNNetworkHelper::getBlob(parent, "custom");
            } else {
                THROW_IE_EXCEPTION << "Unexpected weights layer " << parent->type << " " << parent->name << " for " << layer.type << " " << layer.name;
            }
        } else {
            THROW_IE_EXCEPTION << "Unexpected weights layer type " << weightsLayer->type;
        }
    } else {
        if (layer.blobs.find("weights") == layer.blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (layer.blobs.find('weights') == layer.blobs.end()) {" << std::endl;
            THROW_IE_EXCEPTION << "Convolution weights are absent";
        }
        return layer.blobs.find("weights")->second;
    }
}

Blob::Ptr CNNNetworkHelper::getBiases(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  Blob::Ptr CNNNetworkHelper::getBiases(const CNNLayer& layer) {" << std::endl;
    if (layer.insData.size() > 1U) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (layer.insData.size() > 1U) {" << std::endl;
        if (layer.insData.size() > 2U) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (layer.insData.size() > 2U) {" << std::endl;
            CNNLayerPtr biasesLayer = CNNNetworkHelper::getParent(layer, 2U);
            if (biasesLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (biasesLayer == nullptr) {" << std::endl;
                return nullptr;
            }

            CNNNetworkHelper::checkConstWithBlobs(biasesLayer);
            return biasesLayer->blobs.find("custom")->second;
        } else {
            return nullptr;
        }
    } else {
        const auto it = layer.blobs.find("biases");
        return (it != layer.blobs.end()) ? it->second : nullptr;
    }
}

Blob::Ptr CNNNetworkHelper::quantizeWeights(const CNNLayer& quantize, const bool roundValues, const Precision precision,
                                            const std::vector<float>& weightsShiftPerChannel) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                                              const std::vector<float>& weightsShiftPerChannel) {" << std::endl;
    if (quantize.insData.size() != 5lu) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (quantize.insData.size() != 5lu) {" << std::endl;
        THROW_IE_EXCEPTION << "Unexpected inputs count: " << quantize.insData.size();
    }
    for (int i = 0; i < quantize.insData.size(); i++)
        if (quantize.insData[i].lock() == nullptr)
            THROW_IE_EXCEPTION << "Invalid input data for layer '" << quantize.name << "' with index " << i;

    const Blob::Ptr sourceBlob = getQuantizeLayerBlob(quantize);
    if (sourceBlob == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (sourceBlob == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "weights blob is empty for " << quantize.type << " layer " << quantize.name;
    }

    const Precision blobPrecision = sourceBlob->getTensorDesc().getPrecision();
    if ((precision == Precision::FP32) ||
        ((precision == Precision::UNSPECIFIED) && (blobPrecision == Precision::FP32))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          ((precision == Precision::UNSPECIFIED) && (blobPrecision == Precision::FP32))) {" << std::endl;
        return quantizeBlob<PrecisionTrait<Precision::FP32>::value_type>(quantize, roundValues, precision,
                                                                         weightsShiftPerChannel);
    } else if ((precision == Precision::FP16) ||
               ((precision == Precision::UNSPECIFIED) && (blobPrecision == Precision::FP16))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                 ((precision == Precision::UNSPECIFIED) && (blobPrecision == Precision::FP16))) {" << std::endl;
        return quantizeBlob<PrecisionTrait<Precision::FP16>::value_type>(quantize, roundValues, precision,
                                                                         weightsShiftPerChannel);
    } else if ((precision == Precision::I8) ||
               ((precision == Precision::UNSPECIFIED) && (blobPrecision == Precision::I8))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                 ((precision == Precision::UNSPECIFIED) && (blobPrecision == Precision::I8))) {" << std::endl;
        return quantizeBlob<PrecisionTrait<Precision::I8>::value_type>(quantize, roundValues, precision,
                                                                       weightsShiftPerChannel);
    } else if ((precision == Precision::U8) ||
               ((precision == Precision::UNSPECIFIED) && (blobPrecision == Precision::U8))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:                 ((precision == Precision::UNSPECIFIED) && (blobPrecision == Precision::U8))) {" << std::endl;
        return quantizeBlob<PrecisionTrait<Precision::U8>::value_type>(quantize, roundValues, precision,
                                                                       weightsShiftPerChannel);
    } else {
        THROW_IE_EXCEPTION << "Unexpected precision: " << precision;
    }
}

int CNNNetworkHelper::getConstParentBranchID(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  int CNNNetworkHelper::getConstParentBranchID(const CNNLayer& layer) {" << std::endl;
    int constBranchID = -1;
    for (int i = 0; i < layer.insData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (int i = 0; i < layer.insData.size(); i++) {" << std::endl;
        bool allConst = true;
        if (layer.insData[i].lock() == nullptr)
            THROW_IE_EXCEPTION << "Invalid input data for layer '" << layer.name << "' with index " << i;
        auto parent = layer.insData[i].lock()->getCreatorLayer().lock();
        if (!CaselessEq<std::string>()(parent->type, "FakeQuantize")) continue;
        for (auto& p : parent->insData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (auto& p : parent->insData) {" << std::endl;
            if (!CaselessEq<std::string>()(p.lock()->getCreatorLayer().lock()->type, "Const")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (!CaselessEq<std::string>()(p.lock()->getCreatorLayer().lock()->type, 'Const')) {" << std::endl;
                allConst = false;
                break;
            }
        }
        if (allConst) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (allConst) {" << std::endl;
            constBranchID = i;
            break;
        }
    }

    return constBranchID;
}

int CNNNetworkHelper::getFakeQuantizeBranchWithOneChild(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  int CNNNetworkHelper::getFakeQuantizeBranchWithOneChild(const CNNLayer& layer) {" << std::endl;
    int oneChildBranchID = -1;
    for (size_t i = 0ul; i < layer.insData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (size_t i = 0ul; i < layer.insData.size(); i++) {" << std::endl;
        if (layer.insData[i].lock() == nullptr)
            THROW_IE_EXCEPTION << "Invalid input data for layer '" << layer.name << "' with index " << i;
        auto parent = layer.insData[i].lock()->getCreatorLayer().lock();
        if (!CaselessEq<std::string>()(parent->type, "FakeQuantize")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (!CaselessEq<std::string>()(parent->type, 'FakeQuantize')) {" << std::endl;
            continue;
        }

        if ((parent->outData.size() == 1ul) && (parent->outData[0]->getInputTo().size() == 1ul)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if ((parent->outData.size() == 1ul) && (parent->outData[0]->getInputTo().size() == 1ul)) {" << std::endl;
            oneChildBranchID = i;
            break;
        }
    }

    return oneChildBranchID;
}

Precision CNNNetworkHelper::getPrecisionParent(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  Precision CNNNetworkHelper::getPrecisionParent(const CNNLayer& layer) {" << std::endl;
    return getPrecisionParent(layer, 0ul, false);
}

Precision CNNNetworkHelper::getPrecisionParent(const CNNLayer& layer, const size_t parentIndex) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  Precision CNNNetworkHelper::getPrecisionParent(const CNNLayer& layer, const size_t parentIndex) {" << std::endl;
    return getPrecisionParent(layer, parentIndex, true);
}

Precision CNNNetworkHelper::getPrecisionParent(const CNNLayer& layer, const size_t parentIndex, const bool useParentIndex) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  Precision CNNNetworkHelper::getPrecisionParent(const CNNLayer& layer, const size_t parentIndex, const bool useParentIndex) {" << std::endl;
    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(layer);
    if (parents.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (parents.empty()) {" << std::endl;
        THROW_IE_EXCEPTION << "parents for layer " << layer.type << " '" << layer.name << "' are absent";
    }

    if (useParentIndex) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      if (useParentIndex) {" << std::endl;
        DataPtr parentOutData = getOutData(*parents[parentIndex], layer);
        if (parentOutData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parentOutData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION <<
                "parent layer " << parents[parentIndex]->type << " '" << parents[parentIndex]->name <<
                "' output data  was not found for child " << layer.type << " '" << layer.name << "'";
        }
        return parentOutData->getTensorDesc().getPrecision();
    }

    Precision parentOutDataPrecision = Precision::UNSPECIFIED;
    for (CNNLayerPtr parent : parents) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (CNNLayerPtr parent : parents) {" << std::endl;
        DataPtr parentOutData = getOutData(*parent, layer);
        if (parentOutData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parentOutData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION <<
                "parent layer " << parent->type << " '" << parent->name <<
                "' output data  was not found for child " << layer.type << " '" << layer.name << "'";
        }

        if (parentOutDataPrecision == Precision::UNSPECIFIED) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parentOutDataPrecision == Precision::UNSPECIFIED) {" << std::endl;
            parentOutDataPrecision = parentOutData->getTensorDesc().getPrecision();
        } else if (parentOutDataPrecision != parentOutData->getTensorDesc().getPrecision()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          } else if (parentOutDataPrecision != parentOutData->getTensorDesc().getPrecision()) {" << std::endl;
            THROW_IE_EXCEPTION <<
                "Parent layer " << parent->type << " '" << parent->name <<
                "' output port has unexpected precision " << parentOutData->getTensorDesc().getPrecision();
        }
    }

    return parentOutDataPrecision;
}

DataPtr CNNNetworkHelper::getOutData(const CNNLayer& parentLayer, const CNNLayer& childLayer) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:  DataPtr CNNNetworkHelper::getOutData(const CNNLayer& parentLayer, const CNNLayer& childLayer) {" << std::endl;
    DataPtr parentOutData;
    for (DataPtr outData : parentLayer.outData) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:      for (DataPtr outData : parentLayer.outData) {" << std::endl;
        const std::map<std::string, CNNLayerPtr> inputTo = outData->getInputTo();
        for (auto childIt : inputTo) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          for (auto childIt : inputTo) {" << std::endl;
            if (childIt.second->name == childLayer.name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:              if (childIt.second->name == childLayer.name) {" << std::endl;
                parentOutData = outData;
                break;
            }
        }

        if (parentOutData != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/network_helper.cpp:          if (parentOutData != nullptr) {" << std::endl;
            break;
        }
    }
    return parentOutData;
}
