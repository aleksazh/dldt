#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/const.hpp"
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
#include <vector>

#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void ConstTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/const.cpp:      if (!canBeTransformed(context, layer)) {" << std::endl;
        return;
    }

    if (!CaselessEq<std::string>()(layer.type, "Const")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/const.cpp:      if (!CaselessEq<std::string>()(layer.type, 'Const')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer type '" << layer.name << "' is not correct";
    }

    if ((layer.insData.size() != 0) || (layer.outData.size() != 1)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/const.cpp:      if ((layer.insData.size() != 0) || (layer.outData.size() != 1)) {" << std::endl;
        return;
    }

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    if (!CNNNetworkHelper::IsChild(children, {"FakeQuantize"})) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/const.cpp:      if (!CNNNetworkHelper::IsChild(children, {'FakeQuantize'})) {" << std::endl;
        return;
    }
    if (children.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/const.cpp:      if (children.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "unexpected children count " << children.size();
    }

    const auto fakeQuantize = children[0];
    const CNNLayerPtr inputLayer = CNNNetworkHelper::getParent(*fakeQuantize, 0);
    if (inputLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/const.cpp:      if (inputLayer == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "input data layer for FakeQuantize " << fakeQuantize->name << " is nullable";
    }
    if (inputLayer->name != layer.name) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/const.cpp:      if (inputLayer->name != layer.name) {" << std::endl;
        return;
    }

    const Blob::Ptr weights = CNNNetworkHelper::quantizeWeights(*fakeQuantize, roundQuantizedValues);
    CNNNetworkHelper::transformFakeQuantizeToConst(context, fakeQuantize, weights, layer.name);
}
