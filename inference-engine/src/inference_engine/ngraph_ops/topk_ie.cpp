#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "topk_ie.hpp"

#include <memory>
#include <string>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::TopKIE::type_info;

op::TopKIE::TopKIE(const Output<Node>& data, const Output<Node>& k, const int64_t axis, const std::string& mode, const std::string& sort,
        const Shape& output_shape)
    : Op({data, k}), axis(axis), mode(mode), sort_type(sort), output_shape(output_shape) {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/topk_ie.cpp:      : Op({data, k}), axis(axis), mode(mode), sort_type(sort), output_shape(output_shape) {" << std::endl;
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::TopKIE::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() != 2) {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/topk_ie.cpp:      if (new_args.size() != 2) {" << std::endl;
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<TopKIE>(new_args.at(0), new_args.at(1), axis, mode, sort_type, output_shape);
}

void op::TopKIE::validate_and_infer_types() {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/topk_ie.cpp:  void op::TopKIE::validate_and_infer_types() {" << std::endl;
    set_output_type(0, get_input_element_type(0), output_shape);
    set_output_type(1, element::i32, output_shape);
}

int64_t op::TopKIE::get_axis() {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/topk_ie.cpp:  int64_t op::TopKIE::get_axis() {" << std::endl;
    return axis;
}

std::string op::TopKIE::get_mode() {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/topk_ie.cpp:  std::string op::TopKIE::get_mode() {" << std::endl;
    return mode;
}

std::string op::TopKIE::get_sort_type() {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/topk_ie.cpp:  std::string op::TopKIE::get_sort_type() {" << std::endl;
    return sort_type;
}

Shape op::TopKIE::get_output_shape() {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/topk_ie.cpp:  Shape op::TopKIE::get_output_shape() {" << std::endl;
    return output_shape;
}
