#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pad_ie.hpp"

#include <assert.h>

#include <details/ie_exception.hpp>
#include <memory>
#include <string>
#include <transform/transformations/utils/utils.hpp>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/pad.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PadIE::type_info;

op::PadIE::PadIE(std::shared_ptr<op::v1::Pad> pad)
    : Op({pad->input(0).get_source_output()}),
      m_pad_mode(pad->get_pad_mode()),
      m_pads_begin(pad->get_pads_begin()),
      m_pads_end(pad->get_pads_end()),
      m_output_shape(pad->output(0).get_shape()) {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/pad_ie.cpp:        m_output_shape(pad->output(0).get_shape()) {" << std::endl;
    if (pad->inputs().size() == 4) {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/pad_ie.cpp:      if (pad->inputs().size() == 4) {" << std::endl;
        auto const_node =
            std::dynamic_pointer_cast<op::Constant>(pad->input(3).get_source_output().get_node_shared_ptr());
        if (!const_node) {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/pad_ie.cpp:          if (!const_node) {" << std::endl;
            THROW_IE_EXCEPTION << "Pad " + pad->get_friendly_name() + " with not constant pad_value is not allowed";
        }
        if (!util::get_single_value(const_node, m_pad_value)) {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/pad_ie.cpp:          if (!util::get_single_value(const_node, m_pad_value)) {" << std::endl;
            THROW_IE_EXCEPTION << "Unsupported pad value";
        }
    }
    constructor_validate_and_infer_types();
}

void op::PadIE::validate_and_infer_types() {
    std::cerr << "./inference-engine/src/inference_engine/ngraph_ops/pad_ie.cpp:  void op::PadIE::validate_and_infer_types() {" << std::endl;
    set_output_type(0, get_input_element_type(0), m_output_shape);
}

shared_ptr<Node> op::PadIE::copy_with_new_args(const NodeVector& new_args) const {
    return nullptr;
}
