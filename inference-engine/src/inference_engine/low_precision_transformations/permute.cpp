#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/permute.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void PermuteTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/permute.cpp:      if (!canBeTransformed(context, layer)) {" << std::endl;
        return;
    }

    if (layer.insData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/permute.cpp:      if (layer.insData.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Permute")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/permute.cpp:      if (!CaselessEq<std::string>()(layer.type, 'Permute')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer '" << layer.name << "' is not correct";
    }

    if (!layer.CheckParamPresence("order")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/permute.cpp:      if (!layer.CheckParamPresence('order')) {" << std::endl;
        THROW_IE_EXCEPTION << "Permute parameter 'order' is absent";
    }

    std::vector<unsigned int> orders = layer.GetParamAsUInts("order");
    if ((orders.size() < 2) || (orders[0] != 0U) || (orders[1] != 1U)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/permute.cpp:      if ((orders.size() < 2) || (orders[0] != 0U) || (orders[1] != 1U)) {" << std::endl;
        return;
    }

    TransparentBaseTransformation::transform(context, layer);
}
