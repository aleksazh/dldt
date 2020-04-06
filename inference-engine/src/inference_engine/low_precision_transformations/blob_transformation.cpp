#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/blob_transformation.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <vector>


using namespace InferenceEngine;
using namespace InferenceEngine::details;

void BlobTransformation::transform(ICNNNetwork& network, bool transformWithFakeQuantizeOnWeights) const {
    const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);

    for (const CNNLayerPtr& layer : layers) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/blob_transformation.cpp:      for (const CNNLayerPtr& layer : layers) {" << std::endl;
        if (layer->insData.size() < 2) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/blob_transformation.cpp:          if (layer->insData.size() < 2) {" << std::endl;
            continue;
        }
        if (this->layersForTransformations.find(layer->type) == this->layersForTransformations.end()) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/blob_transformation.cpp:          if (this->layersForTransformations.find(layer->type) == this->layersForTransformations.end()) {" << std::endl;
            continue;
        }

        const CNNLayerPtr weightsLayer = CNNNetworkHelper::getParent(*layer, 1);
        if ((!transformWithFakeQuantizeOnWeights) &&
            ((weightsLayer->type == "FakeQuantize") || (weightsLayer->type == "Quantize"))) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/blob_transformation.cpp:              ((weightsLayer->type == 'FakeQuantize') || (weightsLayer->type == 'Quantize'))) {" << std::endl;
            continue;
        }

        WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(layer.get());
        if (weightableLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/blob_transformation.cpp:          if (weightableLayer == nullptr) {" << std::endl;
            continue;
        }

        const Blob::Ptr weightsBlob = CNNNetworkHelper::getWeights(*layer, false);
        if (weightsBlob != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/blob_transformation.cpp:          if (weightsBlob != nullptr) {" << std::endl;
            weightableLayer->blobs["weights"] = weightsBlob;
            weightableLayer->_weights = weightsBlob;
        }

        if (layer->insData.size() >= 3) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/blob_transformation.cpp:          if (layer->insData.size() >= 3) {" << std::endl;
            const Blob::Ptr biasesBlob = CNNNetworkHelper::getBiases(*layer);
            if (biasesBlob != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/blob_transformation.cpp:              if (biasesBlob != nullptr) {" << std::endl;
                weightableLayer->blobs["biases"] = biasesBlob;
                weightableLayer->_biases = biasesBlob;
            }

            CNNLayerPtr biasesLayer = CNNNetworkHelper::getParent(*layer, 2);
            CNNNetworkHelper::removeLayer(network, biasesLayer);
        }

        CNNNetworkHelper::removeLayer(network, weightsLayer);
    }
}
