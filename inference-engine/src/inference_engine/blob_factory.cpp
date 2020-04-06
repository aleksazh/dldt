#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_factory.hpp"

#include <memory>

InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc) {
    std::cerr << "./inference-engine/src/inference_engine/blob_factory.cpp:  InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc) {" << std::endl;
    return make_blob_with_precision(desc.getPrecision(), desc);
}

InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc, void* ptr) {
    std::cerr << "./inference-engine/src/inference_engine/blob_factory.cpp:  InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc, void* ptr) {" << std::endl;
    return make_blob_with_precision(desc.getPrecision(), desc, ptr);
}

InferenceEngine::Blob::Ptr make_blob_with_precision(const InferenceEngine::TensorDesc& desc,
                                                    const std::shared_ptr<InferenceEngine::IAllocator>& alloc) {
    std::cerr << "./inference-engine/src/inference_engine/blob_factory.cpp:                                                      const std::shared_ptr<InferenceEngine::IAllocator>& alloc) {" << std::endl;
    return make_blob_with_precision(desc.getPrecision(), desc, alloc);
}

InferenceEngine::Blob::Ptr make_plain_blob(InferenceEngine::Precision prec, const InferenceEngine::SizeVector dims) {
    std::cerr << "./inference-engine/src/inference_engine/blob_factory.cpp:  InferenceEngine::Blob::Ptr make_plain_blob(InferenceEngine::Precision prec, const InferenceEngine::SizeVector dims) {" << std::endl;
    return make_blob_with_precision({prec, dims, InferenceEngine::TensorDesc::getLayoutByDims(dims)});
}

InferenceEngine::Blob::Ptr CreateBlobFromData(const InferenceEngine::DataPtr& data) {
    std::cerr << "./inference-engine/src/inference_engine/blob_factory.cpp:  InferenceEngine::Blob::Ptr CreateBlobFromData(const InferenceEngine::DataPtr& data) {" << std::endl;
    // TODO Here some decision should be made about the layout.
    // For now we just pass the layout and use conversion to NCHW for ANY.
    InferenceEngine::Layout targetLayout = data->getLayout();
    if (data->getLayout() == InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/inference_engine/blob_factory.cpp:      if (data->getLayout() == InferenceEngine::Layout::ANY) {" << std::endl;
        targetLayout = InferenceEngine::Layout::NCHW;
    }

    InferenceEngine::TensorDesc desc(data->getPrecision(), data->getTensorDesc().getDims(), targetLayout);

    switch (data->getPrecision()) {
    std::cerr << "./inference-engine/src/inference_engine/blob_factory.cpp:      switch (data->getPrecision()) {" << std::endl;
    case InferenceEngine::Precision::FP32:
        return std::make_shared<InferenceEngine::TBlob<float>>(desc);
    case InferenceEngine::Precision::Q78:
    case InferenceEngine::Precision::I16:
    case InferenceEngine::Precision::FP16:
        return std::make_shared<InferenceEngine::TBlob<short>>(desc);
    case InferenceEngine::Precision::U8:
        return std::make_shared<InferenceEngine::TBlob<uint8_t>>(desc);
    case InferenceEngine::Precision::I8:
        return std::make_shared<InferenceEngine::TBlob<int8_t>>(desc);
    case InferenceEngine::Precision::I32:
        return std::make_shared<InferenceEngine::TBlob<int32_t>>(desc);
    default:
        THROW_IE_EXCEPTION << "precision is no set";
    }
}
