#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_parameter.hpp>
#include <memory>

#if defined(ENABLE_NGRAPH)

#include <ngraph/variant.hpp>

namespace ngraph {

template <>
class VariantWrapper<InferenceEngine::Parameter> : public VariantImpl<InferenceEngine::Parameter> {
public:
    static constexpr VariantTypeInfo type_info {"Variant::InferenceEngine::Parameter", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value): VariantImpl<value_type>(value) {
    std::cerr << "./inference-engine/src/inference_engine/ie_parameter.cpp:      VariantWrapper(const value_type& value): VariantImpl<value_type>(value) {" << std::endl;}  // NOLINT
};

}  // namespace ngraph

constexpr ngraph::VariantTypeInfo ngraph::VariantWrapper<InferenceEngine::Parameter>::type_info;

InferenceEngine::Parameter::Parameter(const std::shared_ptr<ngraph::Variant>& var) {
    std::cerr << "./inference-engine/src/inference_engine/ie_parameter.cpp:  InferenceEngine::Parameter::Parameter(const std::shared_ptr<ngraph::Variant>& var) {" << std::endl;
    if (auto paramWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<InferenceEngine::Parameter>>(var)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_parameter.cpp:      if (auto paramWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<InferenceEngine::Parameter>>(var)) {" << std::endl;
        auto param = paramWrapper->get();
        if (!param.empty()) ptr = param.ptr->copy();
    }
}

InferenceEngine::Parameter::Parameter(std::shared_ptr<ngraph::Variant>& var) {
    std::cerr << "./inference-engine/src/inference_engine/ie_parameter.cpp:  InferenceEngine::Parameter::Parameter(std::shared_ptr<ngraph::Variant>& var) {" << std::endl;
    if (auto paramWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<InferenceEngine::Parameter>>(var)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_parameter.cpp:      if (auto paramWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<InferenceEngine::Parameter>>(var)) {" << std::endl;
        auto param = paramWrapper->get();
        if (!param.empty()) ptr = param.ptr->copy();
    }
}


std::shared_ptr<ngraph::Variant> InferenceEngine::Parameter::asVariant() const {
    return std::make_shared<ngraph::VariantWrapper<InferenceEngine::Parameter>>(*this);
}
#else
InferenceEngine::Parameter::Parameter(const std::shared_ptr<ngraph::Variant>& var) {
    std::cerr << "./inference-engine/src/inference_engine/ie_parameter.cpp:  InferenceEngine::Parameter::Parameter(const std::shared_ptr<ngraph::Variant>& var) {" << std::endl;}
InferenceEngine::Parameter::Parameter(std::shared_ptr<ngraph::Variant>& var) {
    std::cerr << "./inference-engine/src/inference_engine/ie_parameter.cpp:  InferenceEngine::Parameter::Parameter(std::shared_ptr<ngraph::Variant>& var) {" << std::endl;}

std::shared_ptr<ngraph::Variant> InferenceEngine::Parameter::asVariant() const {
    return nullptr;
}
#endif
