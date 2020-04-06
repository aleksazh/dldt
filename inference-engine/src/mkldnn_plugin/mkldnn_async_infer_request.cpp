#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_async_infer_request.h"
#include <memory>

MKLDNNPlugin::MKLDNNAsyncInferRequest::MKLDNNAsyncInferRequest(const InferenceEngine::InferRequestInternal::Ptr &inferRequest,
                                                               const InferenceEngine::ITaskExecutor::Ptr &taskExecutor,
                                                               const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor)
        : InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_async_infer_request.cpp:          : InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor) {" << std::endl;}

void MKLDNNPlugin::MKLDNNAsyncInferRequest::Infer_ThreadUnsafe() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_async_infer_request.cpp:  void MKLDNNPlugin::MKLDNNAsyncInferRequest::Infer_ThreadUnsafe() {" << std::endl;
    InferUsingAsync();
}

MKLDNNPlugin::MKLDNNAsyncInferRequest::~MKLDNNAsyncInferRequest() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_async_infer_request.cpp:  MKLDNNPlugin::MKLDNNAsyncInferRequest::~MKLDNNAsyncInferRequest() {" << std::endl;
    StopAndWait();
}
