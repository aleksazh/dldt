#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "constant_eltwise_reduction.hpp"

#include <ie_ngraph_utils.hpp>
#include <ie_precision.hpp>
#include <memory>
#include <ngraph_ops/scaleshift.hpp>
#include <vector>

#include "ngraph/graph_util.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/pattern/matcher.hpp"

template <typename T>
std::shared_ptr<ngraph::op::Constant> constant_reduction(const std::shared_ptr<ngraph::op::Constant>& const_node) {
    std::cerr << "./inference-engine/src/inference_engine/transform/transformations/constant_eltwise_reduction.cpp:  std::shared_ptr<ngraph::op::Constant> constant_reduction(const std::shared_ptr<ngraph::op::Constant>& const_node) {" << std::endl;
    std::vector<T> data = const_node->get_vector<T>();
    // TODO: implement this function after eltwise broadcast support will be added
    return nullptr;
}

ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
    std::cerr << "./inference-engine/src/inference_engine/transform/transformations/constant_eltwise_reduction.cpp:  ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {" << std::endl;
    // Check that eltwise operation either Add or Multiply
    auto eltwise_node = m.get_match_root();
    if (!std::dynamic_pointer_cast<ngraph::op::v1::Add>(eltwise_node) &&
        !std::dynamic_pointer_cast<ngraph::op::v1::Multiply>(eltwise_node)) {
    std::cerr << "./inference-engine/src/inference_engine/transform/transformations/constant_eltwise_reduction.cpp:          !std::dynamic_pointer_cast<ngraph::op::v1::Multiply>(eltwise_node)) {" << std::endl;
        return false;
    }

    for (const auto& input : eltwise_node->get_inputs()) {
    std::cerr << "./inference-engine/src/inference_engine/transform/transformations/constant_eltwise_reduction.cpp:      for (const auto& input : eltwise_node->get_inputs()) {" << std::endl;
        const auto& inputLayer = input.get_output().get_node();
        auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(inputLayer);
        if (!const_node) continue;

        std::shared_ptr<ngraph::op::Constant> result = nullptr;

        InferenceEngine::Precision ie_precision =
            InferenceEngine::details::ngraph::convertPrecision(const_node->get_element_type());
        switch (ie_precision) {
    std::cerr << "./inference-engine/src/inference_engine/transform/transformations/constant_eltwise_reduction.cpp:          switch (ie_precision) {" << std::endl;
        case InferenceEngine::Precision::FP32:
            result = constant_reduction<float>(const_node);
            break;
        case InferenceEngine::Precision::Q78:
        case InferenceEngine::Precision::I16:
        case InferenceEngine::Precision::FP16:
            result = constant_reduction<short>(const_node);
            break;
        case InferenceEngine::Precision::U8:
            result = constant_reduction<uint8_t>(const_node);
            break;
        case InferenceEngine::Precision::I8:
            result = constant_reduction<int8_t>(const_node);
            break;
        case InferenceEngine::Precision::I32:
            result = constant_reduction<int32_t>(const_node);
            break;
        default:
            return false;
        }
        if (result) {
    std::cerr << "./inference-engine/src/inference_engine/transform/transformations/constant_eltwise_reduction.cpp:          if (result) {" << std::endl;
            ngraph::replace_node(inputLayer, std::dynamic_pointer_cast<ngraph::Node>(result));
            std::cout << "Successful constant_eltwise reduction" << std::endl;
        }
    }

    return true;
};

void ngraph::pass::ConstantEltwiseReduction::constant_multiply_reduction() {
    std::cerr << "./inference-engine/src/inference_engine/transform/transformations/constant_eltwise_reduction.cpp:  void ngraph::pass::ConstantEltwiseReduction::constant_multiply_reduction() {" << std::endl;
    Shape shape {2, 2, 1, 1};
    auto constant1 = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto constant2 = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto eltwise = std::make_shared<ngraph::op::v1::Multiply>(constant1, constant2);

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, "CPUFusion.ConstantMultiplyReduction");
    this->add_matcher(m, callback);
}

void ngraph::pass::ConstantEltwiseReduction::constant_add_reduction() {
    std::cerr << "./inference-engine/src/inference_engine/transform/transformations/constant_eltwise_reduction.cpp:  void ngraph::pass::ConstantEltwiseReduction::constant_add_reduction() {" << std::endl;
    Shape shape {2, 2, 1, 1};
    auto constant1 = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto constant2 = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto eltwise = std::make_shared<ngraph::op::v1::Add>(constant1, constant2);

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, "CPUFusion.ConstantAddReduction");
    this->add_matcher(m, callback);
}
