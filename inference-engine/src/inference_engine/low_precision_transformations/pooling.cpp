#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/pooling.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <string>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void PoolingTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/pooling.cpp:      if (!canBeTransformed(context, layer)) {" << std::endl;
        return;
    }

    if (layer.insData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/pooling.cpp:      if (layer.insData.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Pooling")) {
    std::cerr << "./inference-engine/src/inference_engine/low_precision_transformations/pooling.cpp:      if (!CaselessEq<std::string>()(layer.type, 'Pooling')) {" << std::endl;
        THROW_IE_EXCEPTION << "layer '" << layer.name << "' is not correct";
    }

    TransparentBaseTransformation::transform(context, layer);
}

bool PoolingTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    const std::string poolMethod = layer.GetParamAsString("pool-method", "");
    return poolMethod == "max";
}
