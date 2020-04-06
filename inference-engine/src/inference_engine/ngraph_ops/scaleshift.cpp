#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaleshift.hpp"

#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ScaleShiftIE::type_info;

op::ScaleShiftIE::ScaleShiftIE(const Output<Node>& data_batch, const Output<Node>& weights, const Output<Node>& bias)
    : Op(OutputVector {data_batch, weights, bias}) {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/scaleshift.cpp:      : Op(OutputVector {data_batch, weights, bias}) {" << std::endl;
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::ScaleShiftIE::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() != 3) {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/scaleshift.cpp:      if (new_args.size() != 3) {" << std::endl;
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<ScaleShiftIE>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::ScaleShiftIE::validate_and_infer_types() {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/scaleshift.cpp:  void op::ScaleShiftIE::validate_and_infer_types() {" << std::endl;
    //  Check that weights and biases has the same type
    element::Type data_et = get_input_element_type(0);
    element::Type weights_et = get_input_element_type(1);
    element::Type biases_et = get_input_element_type(2);

    element::Type et_result;
    NODE_VALIDATION_CHECK(this, element::Type::merge(et_result, weights_et, biases_et),
                          "Element types for bias and weights do not match (biases element type: ", biases_et,
                          ", weights element type: ", weights_et, ").");

    set_output_type(0, data_et, get_input_partial_shape(0));
}
