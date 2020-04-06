#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_const_infer_impl.hpp"

#include <map>
#include <string>
#include <vector>

#include "ie_profiling.hpp"

using namespace InferenceEngine;
using namespace ShapeInfer;

void ConstInferImpl::infer(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                           const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/const_infer/ie_const_infer_impl.cpp:                             const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) {" << std::endl;
    std::string errorPrefix = "Ref infer error for Layer with `" + _type + "` type: ";
    if (outData.empty()) THROW_IE_EXCEPTION << errorPrefix + "output data is empty";
    for (auto const& data : outData) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/const_infer/ie_const_infer_impl.cpp:      for (auto const& data : outData) {" << std::endl;
        if (data->buffer() == nullptr) THROW_IE_EXCEPTION << errorPrefix + "output data is not allocated";
    }
    // TODO: check for direct (NCHW, NCH, NC) and FP32
    inferImpl(inData, params, blobs, outData);
}
