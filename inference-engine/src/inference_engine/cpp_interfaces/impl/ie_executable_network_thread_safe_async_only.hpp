// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <iostream>
#include <ie_iinfer_request.hpp>
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp"
#include "cpp_interfaces/base/ie_infer_async_request_base.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_async_request_internal.hpp"

namespace InferenceEngine {

class ExecutableNetworkThreadSafeAsyncOnly
        : public ExecutableNetworkInternal, public std::enable_shared_from_this<ExecutableNetworkThreadSafeAsyncOnly> {
public:
    typedef std::shared_ptr<ExecutableNetworkThreadSafeAsyncOnly> Ptr;

    virtual AsyncInferRequestInternal::Ptr
    CreateAsyncInferRequestImpl(InputsDataMap networkInputs, OutputsDataMap networkOutputs) = 0;

    void CreateInferRequest(IInferRequest::Ptr &asyncRequest) override {
        std::cerr << "ie_executable_network_thread_safe_default.hpp CreateInferRequest auto asyncRequestImpl = ..." << std::endl;
        auto asyncRequestImpl = this->CreateAsyncInferRequestImpl(_networkInputs, _networkOutputs);
        std::cerr << "ie_executable_network_thread_safe_default.hpp CreateInferRequest asyncRequestImpl->setPointerToExecutableNetworkInternal( ..." << std::endl;
        asyncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        std::cerr << "ie_executable_network_thread_safe_default.hpp CreateInferRequest asyncRequest.reset(new ..." << std::endl;
        asyncRequest.reset(new InferRequestBase<AsyncInferRequestInternal>(asyncRequestImpl),
                           [](IInferRequest *p) { p->Release(); });
        std::cerr << "ie_executable_network_thread_safe_default.hpp CreateInferRequest asyncRequestImpl->SetPublicInterfacePtr( ..." << std::endl;
        asyncRequestImpl->SetPublicInterfacePtr(asyncRequest);
    }
};

}  // namespace InferenceEngine
