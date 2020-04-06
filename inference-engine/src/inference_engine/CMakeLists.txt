# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine")

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(ngraph_src
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/assertion.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/attribute_adapter.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/attribute_adapter.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/attribute_visitor.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/autodiff/adjoints.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/autodiff/adjoints.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/axis_set.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/axis_set.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/axis_vector.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/axis_vector.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/autobroadcast.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/autobroadcast.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/dequantize_builder.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/dequantize_builder.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/make_constant.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/matmul_factory.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/matmul_factory.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/norm.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/norm.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/numpy_transpose.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/numpy_transpose.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantize_builder.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantize_builder.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantized_concat_builder.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantized_concat_builder.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantized_conv_builder.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantized_conv_builder.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantized_dot_builder.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantized_dot_builder.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantization/quantized_linear_convolution.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantization/quantized_linear_convolution.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantization_utils.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/quantization_utils.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/reduce_ops.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/reduce_ops.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/reshape.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/reshape.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/split.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/split.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/builder/tensor_mask.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/check.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/chrome_trace.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/chrome_trace.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/code_writer.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/component_manager.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/coordinate.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/coordinate.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/coordinate_diff.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/coordinate_diff.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/coordinate_transform.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/cpio.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/cpio.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/deprecated.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/input.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/input.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/layout/dense_tensor_layout.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/layout/dense_tensor_layout.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/layout/tensor_layout.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/layout/tensor_layout.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/output.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/output.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/tensor.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/descriptor/tensor.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/dimension.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/dimension.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/distributed/null.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/distributed/null.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/distributed.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/distributed.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/enum_names.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/env_util.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/env_util.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/except.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/factory.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/factory.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/file_util.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/file_util.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/function.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/function.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/graph_util.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/lambda.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/lambda.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/log.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/log.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/ngraph.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/ngraph.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/ngraph_visibility.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/node_input.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/node_input.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/node_output.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/node_output.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/node.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/node.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/abs.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/abs.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/acos.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/acos.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/add.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/add.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/all.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/all.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/allreduce.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/allreduce.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/and.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/and.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/any.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/any.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/argmax.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/argmax.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/argmin.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/argmin.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/asin.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/asin.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/atan.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/atan.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/atan2.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/atan2.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/avg_pool.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/avg_pool.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/batch_norm.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/batch_norm.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/binary_convolution.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/binary_convolution.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/broadcast.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/broadcast_distributed.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/ceiling.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/ceiling.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/concat.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/concat.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/constant.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/constant.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/convert.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/convert.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/convert_like.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/convert_like.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/convolution.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/convolution.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/cos.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/cos.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/cosh.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/cosh.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/cum_sum.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/cum_sum.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/crop_and_resize.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/crop_and_resize.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/deformable_convolution.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/deformable_convolution.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/deformable_psroi_pooling.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/deformable_psroi_pooling.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/dequantize.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/dequantize.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/divide.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/divide.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/dot.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/dot.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/embedding_lookup.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/embedding_lookup.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/equal.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/equal.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/erf.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/erf.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/exp.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/exp.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/batch_mat_mul.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/batch_mat_mul.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_broadcast.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_broadcast.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_pad.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_pad.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_replace_slice.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_replace_slice.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_reshape.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_reshape.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_slice.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/dyn_slice.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/generate_mask.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/generate_mask.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/quantized_conv_bias.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/quantized_conv_bias.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/quantized_conv_relu.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/quantized_conv_relu.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/range.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/range.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/shape_of.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/shape_of.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/tile.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/tile.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/quantized_dot_bias.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/quantized_dot_bias.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/compiled_kernel.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/compiled_kernel.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/transpose.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/transpose.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/ctc_greedy_decoder.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/ctc_greedy_decoder.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/detection_output.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/detection_output.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/interpolate.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/interpolate.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/prior_box.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/prior_box.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/prior_box_clustered.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/prior_box_clustered.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/proposal.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/proposal.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/psroi_pooling.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/psroi_pooling.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/region_yolo.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/region_yolo.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/reorg_yolo.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/reorg_yolo.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/roi_pooling.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/layers/roi_pooling.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/random_uniform.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/experimental/random_uniform.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/strided_slice.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/floor.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/floor.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/floor_mod.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/floor_mod.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/gather.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/gather.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/gather_nd.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/gather_nd.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/gather_tree.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/gather_tree.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/get_output_element.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/get_output_element.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/greater.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/greater.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/greater_eq.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/greater_eq.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/less.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/less.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/less_eq.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/less_eq.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/log.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/log.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/lrn.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/lrn.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/max.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/max.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/max_pool.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/max_pool.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/maximum.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/maximum.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/min.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/min.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/minimum.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/minimum.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/multiply.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/multiply.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/negative.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/negative.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/non_max_suppression.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/non_max_suppression.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/not.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/not.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/not_equal.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/not_equal.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/one_hot.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/one_hot.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/op.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/op.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/or.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/or.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/pad.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/pad.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/parameter.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/parameter.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/passthrough.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/passthrough.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/power.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/power.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/product.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/product.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_logical_and.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_logical_and.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_logical_or.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_logical_or.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_prod.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_prod.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_mean.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_mean.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_sum.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reduce_sum.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/round.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/round.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/quantize.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/quantize.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/quantized_convolution.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/quantized_convolution.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/quantized_dot.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/quantized_dot.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/recv.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/recv.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/relu.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/relu.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/replace_slice.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/replace_slice.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reshape.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reshape.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/result.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/result.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reverse.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reverse.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reverse_sequence.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/reverse_sequence.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/scatter_add.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/scatter_add.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/scatter_nd_add.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/scatter_nd_add.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/select.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/select.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/send.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/send.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sigmoid.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sigmoid.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sign.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sign.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sin.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sin.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sinh.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sinh.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/slice.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/slice.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/softmax.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/softmax.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/strided_slice.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/strided_slice.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sqrt.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sqrt.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/stop_gradient.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/stop_gradient.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/subtract.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/subtract.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sum.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/sum.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/variadic_split.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/variadic_split.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/tan.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/tan.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/tanh.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/tanh.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/tensor_iterator.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/tensor_iterator.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/topk.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/topk.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/xor.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/xor.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/batch_mat_mul_transpose.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/batch_mat_mul_transpose.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/batch_to_space.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/batch_to_space.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/clamp.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/clamp.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/conv_fused.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/conv_fused.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/crossentropy.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/crossentropy.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/hard_sigmoid.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/hard_sigmoid.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/depth_to_space.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/depth_to_space.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/elu.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/elu.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/fake_quantize.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/fake_quantize.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/gelu.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/gelu.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/gemm.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/gemm.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/grn.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/grn.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/group_conv.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/group_conv.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/gru_cell.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/gru_cell.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/layer_norm.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/layer_norm.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/lstm_cell.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/lstm_cell.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/lstm_sequence.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/lstm_sequence.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/matmul.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/matmul.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/mod.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/mod.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/mvn.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/mvn.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/normalize_l2.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/normalize_l2.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/partial_slice.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/partial_slice.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/prelu.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/prelu.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/rnn_cell.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/rnn_cell.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/scale_shift.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/scale_shift.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/scatter_nd.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/scatter_nd.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/stack.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/stack.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/selu.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/selu.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/shuffle_channels.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/shuffle_channels.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/softmax_crossentropy.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/softmax_crossentropy.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/space_to_batch.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/space_to_batch.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/space_to_depth.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/space_to_depth.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/split.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/split.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/squared_difference.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/squared_difference.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/squeeze.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/squeeze.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/unsqueeze.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/fused/unsqueeze.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/attr_types.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/attr_types.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/activation_functions.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/activation_functions.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/arithmetic_reduction.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/arithmetic_reduction.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/arithmetic_reductions_keep_dims.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/arithmetic_reductions_keep_dims.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/binary_elementwise_arithmetic.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/binary_elementwise_arithmetic.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/binary_elementwise_comparison.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/binary_elementwise_comparison.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/binary_elementwise_logical.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/binary_elementwise_logical.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/fused_op.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/fused_op.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/index_reduction.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/index_reduction.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/logical_reduction_keep_dims.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/logical_reduction_keep_dims.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/logical_reduction.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/logical_reduction.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/rnn_cell_base.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/rnn_cell_base.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/unary_elementwise_arithmetic.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/op/util/unary_elementwise_arithmetic.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/ops.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/opsets/opset.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/partial_shape.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/partial_shape.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/algebraic_simplification.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/algebraic_simplification.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/assign_layout.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/implicit_broadcast_elimination.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/implicit_broadcast_elimination.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/batch_fusion.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/batch_fusion.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/common_function_collection.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/common_function_collection.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_arithmetic_reduction.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_binary.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_broadcast.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_concat.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_convert.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_dequantize.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_dyn_broadcast.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_dyn_reshape.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_dyn_slice.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_gather.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_logical_reduction.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_one_hot.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_pad.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_quantize.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_range.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_reshape.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_reverse.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_select.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_squeeze.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_unsqueeze.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_shape_of.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_slice.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_split.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_variadic_split.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_strided_slice.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_tile.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_transpose.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding_unary.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_folding.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/constant_to_broadcast.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/convert_fp32_to_fp16.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/convert_fp32_to_fp16.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/core_fusion.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/core_fusion.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/cse.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/cse.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/dump_sorted.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/dump_sorted.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/dyn_elimination.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/dyn_elimination.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/fused_op_decomposition.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/fused_op_decomposition.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/get_output_element_elimination.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/get_output_element_elimination.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/graph_rewrite.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/graph_rewrite.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/like_replacement.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/like_replacement.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/liveness.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/liveness.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/manager.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/manager.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/manager_state.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/memory_layout.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/memory_layout.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/memory_visualize.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/memory_visualize.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/nop_elimination.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/nop_elimination.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/pass.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/pass.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/opset0_downgrade.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/opset0_downgrade.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/opset1_upgrade.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/opset1_upgrade.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/pass_config.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/pass_config.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/propagate_cacheability.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/propagate_cacheability.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/reshape_elimination.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/reshape_elimination.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/reshape_sinking.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/reshape_sinking.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/serialize.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/serialize.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/shape_relevance.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/shape_relevance.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/validate_graph.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/validate_graph.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/validate.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/validate.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/visualize_tree.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/visualize_tree.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/zero_dim_tensor_elimination.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/zero_dim_tensor_elimination.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/zero_dim_tensor_elimination.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/zero_dim_tensor_elimination.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/concat_fusion.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/concat_fusion.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/pass_util.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pass/pass_util.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/matcher.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/matcher.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/any.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/any.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/any_of.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/any_of.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/branch.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/branch.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/capture.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/capture.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/label.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/label.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/or.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/or.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/pattern.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/pattern.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/skip.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/skip.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/true.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/pattern/op/true.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/placement.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/placement.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/provenance.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/provenance.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/rank.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/aligned_buffer.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/aligned_buffer.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/allocator.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/allocator.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/backend.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/backend.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/backend_manager.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/backend_manager.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/cache.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/cache.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/executable.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/executable.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/host_tensor.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/host_tensor.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/performance_counter.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/tensor.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/tensor.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/shape.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/shape.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/shape_util.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/shape_util.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/slice_plan.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/slice_plan.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/specialize_function.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/specialize_function.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/state/bernoulli_rng_state.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/state/bernoulli_rng_state.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/state/uniform_rng_state.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/state/uniform_rng_state.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/strides.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/strides.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/type/bfloat16.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/type/bfloat16.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/type/float16.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/type/float16.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/type/element_type.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/type.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/type.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/util.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/util.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/validation_util.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/validation_util.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/variant.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/variant.hpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/dynamic/dynamic_backend.cpp
${OpenVINO_MAIN_SOURCE_DIR}/ngraph/src/ngraph/runtime/dynamic/dynamic_backend.hpp

${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/thirdparty/pugixml/src/pugixml.cpp
${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/thirdparty/pugixml/src/pugiconfig.hpp
${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/thirdparty/pugixml/src/pugixml.hpp

)

set(builder_files_src ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_layer_decorator.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_layer_builder.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_input_layer_layer.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_const_layer.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_split_layer.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_pooling_layer.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/builders/ie_network_builder.cpp)
#set(builder_files_src ${CMAKE_CURRENT_SOURCE_DIR}/builders/*.cpp)

file (GLOB LIBRARY_SRC
        ${ngraph_src}
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/low_precision_transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/low_precision_transformations/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/transformations/fusion/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/transformations/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_ops/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/built-in/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/const_infer/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/impl/*.cpp
        ${builder_files_src}
      )

if (ENABLE_NGRAPH)
    file (GLOB LIBRARY_SRC ${LIBRARY_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transform/transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_ops/*.cpp)
endif()

file (GLOB LIBRARY_HEADERS
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/built-in/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer/const_infer/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/base/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/impl/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/cpp_interfaces/interface/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/low_precision_transformations/*.hpp
       ${CMAKE_CURRENT_SOURCE_DIR}/low_precision_transformations/common/*.hpp
      )

if (NOT ENABLE_NGRAPH)
    list(REMOVE_ITEM LIBRARY_SRC "${CMAKE_CURRENT_SOURCE_DIR}/ie_cnn_layer_builder_ngraph.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cnn_network_ngraph_impl.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_ir_parser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_ir_reader.cpp")
    list(REMOVE_ITEM LIBRARY_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/ie_ngraph_utils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cnn_network_ngraph_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_cnn_layer_builder_ngraph.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_ir_parser.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ie_ir_reader.hpp")
endif()

if(((NOT DEFINED ENABLE_SSE42) OR ENABLE_SSE42) AND NOT ENABLE_EMSCRIPTEN)
    file (GLOB LIBRARY_SRC
           ${LIBRARY_SRC}
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.cpp
          )
    file (GLOB LIBRARY_HEADERS
           ${LIBRARY_HEADERS}
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.hpp
          )

    file (GLOB SSE_SRC
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.cpp
          )
    file (GLOB SSE_HEADERS
           ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.hpp
          )

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42)

    if (WIN32)
        if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
            # No such option for MSVC 2019
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
            set_source_files_properties(${SSE_SRC}
                PROPERTIES COMPILE_FLAGS /arch:SSE4.2 /QxSSE4.2 /Qvc14)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
            set_source_files_properties(${SSE_SRC}
                PROPERTIES COMPILE_FLAGS -msse4.2)
        endif()
    else()
        set_source_files_properties(${SSE_SRC}
                PROPERTIES COMPILE_FLAGS -msse4.2)
    endif()
    add_definitions(-DHAVE_SSE=1)
endif()

addVersionDefines(ie_version.cpp CI_BUILD_NUMBER)

set (PUBLIC_HEADERS_DIR "${IE_MAIN_SOURCE_DIR}/include")

file (GLOB PUBLIC_HEADERS
       ${PUBLIC_HEADERS_DIR}/builders/*.hpp
       ${PUBLIC_HEADERS_DIR}/builders/*.h
       ${PUBLIC_HEADERS_DIR}/cpp/*.hpp
       ${PUBLIC_HEADERS_DIR}/cpp/*.h
       ${PUBLIC_HEADERS_DIR}/details/os/*.hpp
       ${PUBLIC_HEADERS_DIR}/details/os/*.h
       ${PUBLIC_HEADERS_DIR}/details/*.hpp
       ${PUBLIC_HEADERS_DIR}/details/*.h
       ${PUBLIC_HEADERS_DIR}/*.hpp
       ${PUBLIC_HEADERS_DIR}/*.h
      )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS} ${PUBLIC_HEADERS})

# Plugin API library

add_library(${TARGET_NAME}_plugin_api INTERFACE)

target_include_directories(${TARGET_NAME}_plugin_api INTERFACE
    "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
    $<TARGET_PROPERTY:${TARGET_NAME}_preproc,INTERFACE_INCLUDE_DIRECTORIES>
    ${PUBLIC_HEADERS_DIR})

if(ENABLE_NGRAPH)
    target_compile_definitions(${TARGET_NAME}_plugin_api INTERFACE ENABLE_NGRAPH)
endif()

# Create object library

add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS}
            ${PUBLIC_HEADERS})

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)

if (ENABLE_NGRAPH)
    target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE $<TARGET_PROPERTY:ngraph::ngraph,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src")

target_include_directories(${TARGET_NAME}_obj PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                      $<TARGET_PROPERTY:inference_engine_preproc,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:${TARGET_NAME}_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

if(ENABLE_PROFILING_ITT AND INTEL_ITT_LIBS)
    target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_COMPILE_DEFINITIONS>)
endif()

if(ENABLE_MKL_DNN)
    target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu/xbyak")
endif()

set_ie_threading_interface_for(${TARGET_NAME}_obj)

if(WIN32)
    # To disable min/max macro in windows.h
    target_compile_definitions(${TARGET_NAME}_obj PRIVATE -DNOMINMAX)
endif()

if(ENABLE_IR_READER)
    target_compile_definitions(${TARGET_NAME}_obj PRIVATE ENABLE_IR_READER)
endif()

#add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME}_obj EXCLUDE_PATTERNS "../ngraph/*")
#add_clang_format_target(${TARGET_NAME}_clang_format FOR_TARGETS ${TARGET_NAME}_obj EXCLUDE_PATTERNS "../ngraph/*")

# Create shared library file from object library

add_library(${TARGET_NAME} SHARED
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

set_ie_threading_interface_for(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads ${INTEL_ITT_LIBS} pugixml ${NGRAPH_LIBRARIES} PUBLIC ${CMAKE_DL_LIBS})

target_include_directories(${TARGET_NAME} INTERFACE ${PUBLIC_HEADERS_DIR})

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()


if(ENABLE_UNICODE_PATH_SUPPORT)
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_UNICODE_PATH_SUPPORT)
endif()

ie_register_plugins(MAIN_TARGET ${TARGET_NAME}
                    POSSIBLE_PLUGINS HeteroPlugin clDNNPlugin GNAPlugin MKLDNNPlugin myriadPlugin)

# Create NN Builder

file(GLOB NN_BUILDER_LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/builders/*.cpp)

# disable deprecated warnings for NN Builder

function(nn_builder_disable_warnings)
    disable_deprecated_warnings()
    set_source_files_properties(${NN_BUILDER_LIBRARY_SRC} PROPERTIES COMPILE_FLAGS ${ie_c_cxx_deprecated})
endfunction()

nn_builder_disable_warnings()

list(REMOVE_ITEM NN_BUILDER_LIBRARY_SRC ${builder_files_src})

add_library(${TARGET_NAME}_nn_builder SHARED ${NN_BUILDER_LIBRARY_SRC})

target_compile_definitions(${TARGET_NAME}_nn_builder PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)

target_link_libraries(${TARGET_NAME}_nn_builder PUBLIC ${TARGET_NAME})

target_include_directories(${TARGET_NAME}_nn_builder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>
            ${NN_BUILDER_LIBRARY_SRC})

set_ie_threading_interface_for(${TARGET_NAME}_s)

target_include_directories(${TARGET_NAME}_s PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

if(WIN32)
    set_target_properties(${TARGET_NAME}_s PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_s)
endif()

if(ENABLE_UNICODE_PATH_SUPPORT)
    target_compile_definitions(${TARGET_NAME}_s PUBLIC ENABLE_UNICODE_PATH_SUPPORT)
endif()

target_link_libraries(${TARGET_NAME}_s PRIVATE ${NGRAPH_LIBRARIES}
# NOTE: public linkage with ITT is important here! without it binaries that use this library do not
#       get proper compile flags to support ITT which causes hard-to-detect bugs.
                                       PUBLIC pugixml ${INTEL_ITT_LIBS} ${CMAKE_DL_LIBS})

target_compile_definitions(${TARGET_NAME}_s PUBLIC USE_STATIC_IE)

if(ENABLE_PROFILING_ITT AND INTEL_ITT_LIBS)
    target_include_directories(${TARGET_NAME}_s SYSTEM PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${TARGET_NAME}_s PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_COMPILE_DEFINITIONS>)
endif()

# export targets

export(TARGETS ${TARGET_NAME} ${TARGET_NAME}_nn_builder NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")

configure_file(
    "${IE_MAIN_SOURCE_DIR}/cmake/config.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineConfig.cmake"
    COPYONLY)

configure_file(
    "${IE_MAIN_SOURCE_DIR}/cmake/share/InferenceEngineConfig-version.cmake.in"
    "${CMAKE_BINARY_DIR}/InferenceEngineConfig-version.cmake"
    COPYONLY)

# developer package

add_library(xbyak INTERFACE)
target_include_directories(xbyak INTERFACE ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu/xbyak)

ie_developer_export_targets(${TARGET_NAME} ${TARGET_NAME}_plugin_api ${TARGET_NAME}_nn_builder xbyak)

# install

if(THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    set(threading_tbb ON)
endif()

if(ENABLE_NGRAPH)
    list(APPEND core_components ngraph)
endif()

if(threading_tbb)
    ie_cpack_add_component(tbb REQUIRED)
    list(APPEND core_components tbb)

    install(DIRECTORY "${TBB}/include"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
    install(DIRECTORY "${TBB}/lib"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
    if(EXISTS "${TBB}/bin")
      install(DIRECTORY "${TBB}/bin"
              DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
              COMPONENT tbb)
    endif()
    install(FILES "${TBB}/LICENSE"
            DESTINATION ${IE_CPACK_IE_DIR}/external/tbb
            COMPONENT tbb)
endif()

ie_cpack_add_component(core REQUIRED DEPENDS ${core_components})

install(DIRECTORY "${IE_MAIN_SOURCE_DIR}/include" DESTINATION ${IE_CPACK_IE_DIR}
        COMPONENT core)
install(TARGETS ${TARGET_NAME} ${TARGET_NAME}_nn_builder
        RUNTIME DESTINATION ${IE_CPACK_LIBRARY_PATH}
        ARCHIVE DESTINATION ${IE_CPACK_LIBRARY_PATH}
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH}
        COMPONENT core)
install(FILES "${OpenVINO_BINARY_DIR}/share/ie_parallel.cmake"
              "${OpenVINO_BINARY_DIR}/share/ie_rh_decoder.cmake"
              "${OpenVINO_BINARY_DIR}/share/InferenceEngineConfig.cmake"
              "${OpenVINO_BINARY_DIR}/share/InferenceEngineConfig-version.cmake"
        DESTINATION ${IE_CPACK_IE_DIR}/share
        COMPONENT core)
install(FILES $<TARGET_FILE_DIR:${TARGET_NAME}>/plugins.xml
        DESTINATION ${IE_CPACK_LIBRARY_PATH}
        COMPONENT core)
