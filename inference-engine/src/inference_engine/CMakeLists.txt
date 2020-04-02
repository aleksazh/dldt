# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine")

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(ngraph_src
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/assertion.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/attribute_adapter.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/attribute_adapter.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/attribute_visitor.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/autodiff/adjoints.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/autodiff/adjoints.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/axis_set.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/axis_set.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/axis_vector.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/axis_vector.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/autobroadcast.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/autobroadcast.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/dequantize_builder.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/dequantize_builder.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/make_constant.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/matmul_factory.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/matmul_factory.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/norm.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/norm.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/numpy_transpose.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/numpy_transpose.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantize_builder.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantize_builder.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantized_concat_builder.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantized_concat_builder.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantized_conv_builder.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantized_conv_builder.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantized_dot_builder.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantized_dot_builder.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantization/quantized_linear_convolution.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantization/quantized_linear_convolution.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantization_utils.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/quantization_utils.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/reduce_ops.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/reduce_ops.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/reshape.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/reshape.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/split.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/split.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/builder/tensor_mask.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/check.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/chrome_trace.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/chrome_trace.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/code_writer.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/component_manager.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/coordinate.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/coordinate.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/coordinate_diff.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/coordinate_diff.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/coordinate_transform.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/cpio.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/cpio.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/deprecated.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/input.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/input.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/layout/dense_tensor_layout.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/layout/dense_tensor_layout.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/layout/tensor_layout.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/layout/tensor_layout.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/output.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/output.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/tensor.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/descriptor/tensor.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/dimension.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/dimension.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/distributed/null.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/distributed/null.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/distributed.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/distributed.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/enum_names.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/env_util.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/env_util.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/except.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/factory.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/factory.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/file_util.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/file_util.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/function.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/function.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/graph_util.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/lambda.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/lambda.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/log.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/log.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/ngraph.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/ngraph.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/ngraph_visibility.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/node_input.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/node_input.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/node_output.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/node_output.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/node.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/node.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/abs.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/abs.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/acos.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/acos.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/add.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/add.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/all.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/all.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/allreduce.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/allreduce.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/and.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/and.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/any.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/any.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/argmax.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/argmax.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/argmin.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/argmin.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/asin.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/asin.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/atan.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/atan.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/atan2.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/atan2.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/avg_pool.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/avg_pool.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/batch_norm.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/batch_norm.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/binary_convolution.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/binary_convolution.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/broadcast.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/broadcast_distributed.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/ceiling.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/ceiling.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/concat.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/concat.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/constant.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/constant.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/convert.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/convert.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/convert_like.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/convert_like.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/convolution.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/convolution.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/cos.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/cos.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/cosh.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/cosh.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/cum_sum.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/cum_sum.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/crop_and_resize.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/crop_and_resize.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/deformable_convolution.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/deformable_convolution.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/deformable_psroi_pooling.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/deformable_psroi_pooling.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/dequantize.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/dequantize.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/divide.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/divide.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/dot.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/dot.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/embedding_lookup.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/embedding_lookup.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/equal.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/equal.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/erf.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/erf.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/exp.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/exp.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/batch_mat_mul.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/batch_mat_mul.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_broadcast.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_broadcast.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_pad.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_pad.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_replace_slice.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_replace_slice.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_reshape.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_reshape.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_slice.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/dyn_slice.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/generate_mask.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/generate_mask.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/quantized_conv_bias.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/quantized_conv_bias.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/quantized_conv_relu.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/quantized_conv_relu.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/range.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/range.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/shape_of.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/shape_of.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/tile.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/tile.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/quantized_dot_bias.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/quantized_dot_bias.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/compiled_kernel.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/compiled_kernel.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/transpose.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/transpose.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/ctc_greedy_decoder.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/ctc_greedy_decoder.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/detection_output.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/detection_output.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/interpolate.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/interpolate.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/prior_box.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/prior_box.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/prior_box_clustered.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/prior_box_clustered.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/proposal.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/proposal.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/psroi_pooling.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/psroi_pooling.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/region_yolo.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/region_yolo.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/reorg_yolo.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/reorg_yolo.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/roi_pooling.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/layers/roi_pooling.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/random_uniform.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/experimental/random_uniform.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/strided_slice.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/floor.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/floor.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/floor_mod.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/floor_mod.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/gather.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/gather.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/gather_nd.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/gather_nd.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/gather_tree.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/gather_tree.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/get_output_element.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/get_output_element.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/greater.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/greater.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/greater_eq.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/greater_eq.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/less.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/less.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/less_eq.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/less_eq.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/log.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/log.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/lrn.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/lrn.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/max.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/max.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/max_pool.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/max_pool.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/maximum.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/maximum.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/min.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/min.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/minimum.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/minimum.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/multiply.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/multiply.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/negative.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/negative.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/non_max_suppression.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/non_max_suppression.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/not.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/not.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/not_equal.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/not_equal.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/one_hot.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/one_hot.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/op.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/op.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/or.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/or.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/pad.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/pad.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/parameter.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/parameter.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/passthrough.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/passthrough.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/power.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/power.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/product.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/product.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_logical_and.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_logical_and.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_logical_or.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_logical_or.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_prod.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_prod.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_mean.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_mean.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_sum.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reduce_sum.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/round.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/round.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/quantize.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/quantize.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/quantized_convolution.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/quantized_convolution.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/quantized_dot.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/quantized_dot.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/recv.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/recv.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/relu.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/relu.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/replace_slice.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/replace_slice.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reshape.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reshape.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/result.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/result.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reverse.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reverse.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reverse_sequence.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/reverse_sequence.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/scatter_add.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/scatter_add.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/scatter_nd_add.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/scatter_nd_add.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/select.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/select.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/send.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/send.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sigmoid.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sigmoid.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sign.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sign.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sin.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sin.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sinh.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sinh.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/slice.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/slice.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/softmax.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/softmax.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/strided_slice.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/strided_slice.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sqrt.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sqrt.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/stop_gradient.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/stop_gradient.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/subtract.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/subtract.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sum.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/sum.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/variadic_split.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/variadic_split.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/tan.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/tan.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/tanh.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/tanh.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/tensor_iterator.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/tensor_iterator.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/topk.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/topk.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/xor.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/xor.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/batch_mat_mul_transpose.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/batch_mat_mul_transpose.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/batch_to_space.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/batch_to_space.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/clamp.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/clamp.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/conv_fused.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/conv_fused.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/crossentropy.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/crossentropy.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/hard_sigmoid.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/hard_sigmoid.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/depth_to_space.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/depth_to_space.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/elu.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/elu.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/fake_quantize.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/fake_quantize.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/gelu.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/gelu.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/gemm.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/gemm.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/grn.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/grn.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/group_conv.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/group_conv.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/gru_cell.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/gru_cell.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/layer_norm.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/layer_norm.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/lstm_cell.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/lstm_cell.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/lstm_sequence.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/lstm_sequence.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/matmul.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/matmul.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/mod.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/mod.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/mvn.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/mvn.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/normalize_l2.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/normalize_l2.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/partial_slice.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/partial_slice.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/prelu.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/prelu.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/rnn_cell.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/rnn_cell.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/scale_shift.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/scale_shift.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/scatter_nd.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/scatter_nd.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/stack.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/stack.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/selu.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/selu.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/shuffle_channels.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/shuffle_channels.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/softmax_crossentropy.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/softmax_crossentropy.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/space_to_batch.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/space_to_batch.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/space_to_depth.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/space_to_depth.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/split.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/split.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/squared_difference.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/squared_difference.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/squeeze.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/squeeze.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/unsqueeze.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/fused/unsqueeze.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/attr_types.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/attr_types.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/activation_functions.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/activation_functions.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/arithmetic_reduction.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/arithmetic_reduction.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/arithmetic_reductions_keep_dims.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/arithmetic_reductions_keep_dims.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/binary_elementwise_arithmetic.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/binary_elementwise_arithmetic.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/binary_elementwise_comparison.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/binary_elementwise_comparison.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/binary_elementwise_logical.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/binary_elementwise_logical.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/fused_op.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/fused_op.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/index_reduction.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/index_reduction.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/logical_reduction_keep_dims.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/logical_reduction_keep_dims.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/logical_reduction.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/logical_reduction.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/rnn_cell_base.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/rnn_cell_base.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/unary_elementwise_arithmetic.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/op/util/unary_elementwise_arithmetic.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/ops.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/opsets/opset.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/partial_shape.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/partial_shape.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/algebraic_simplification.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/algebraic_simplification.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/assign_layout.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/implicit_broadcast_elimination.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/implicit_broadcast_elimination.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/batch_fusion.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/batch_fusion.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/common_function_collection.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/common_function_collection.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_arithmetic_reduction.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_binary.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_broadcast.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_concat.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_convert.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_dequantize.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_dyn_broadcast.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_dyn_reshape.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_dyn_slice.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_gather.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_logical_reduction.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_one_hot.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_pad.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_quantize.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_range.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_reshape.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_reverse.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_select.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_squeeze.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_unsqueeze.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_shape_of.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_slice.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_split.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_variadic_split.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_strided_slice.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_tile.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_transpose.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding_unary.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_folding.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/constant_to_broadcast.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/convert_fp32_to_fp16.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/convert_fp32_to_fp16.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/core_fusion.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/core_fusion.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/cse.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/cse.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/dump_sorted.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/dump_sorted.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/dyn_elimination.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/dyn_elimination.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/fused_op_decomposition.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/fused_op_decomposition.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/get_output_element_elimination.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/get_output_element_elimination.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/graph_rewrite.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/graph_rewrite.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/like_replacement.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/like_replacement.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/liveness.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/liveness.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/manager.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/manager.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/manager_state.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/memory_layout.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/memory_layout.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/memory_visualize.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/memory_visualize.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/nop_elimination.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/nop_elimination.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/pass.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/pass.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/opset0_downgrade.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/opset0_downgrade.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/opset1_upgrade.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/opset1_upgrade.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/pass_config.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/pass_config.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/propagate_cacheability.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/propagate_cacheability.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/reshape_elimination.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/reshape_elimination.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/reshape_sinking.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/reshape_sinking.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/serialize.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/serialize.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/shape_relevance.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/shape_relevance.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/validate_graph.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/validate_graph.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/validate.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/validate.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/visualize_tree.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/visualize_tree.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/zero_dim_tensor_elimination.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/zero_dim_tensor_elimination.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/zero_dim_tensor_elimination.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/zero_dim_tensor_elimination.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/concat_fusion.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/concat_fusion.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/pass_util.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pass/pass_util.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/matcher.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/matcher.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/any.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/any.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/any_of.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/any_of.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/branch.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/branch.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/capture.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/capture.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/label.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/label.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/or.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/or.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/pattern.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/pattern.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/skip.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/skip.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/true.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/pattern/op/true.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/placement.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/placement.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/provenance.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/provenance.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/rank.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/aligned_buffer.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/aligned_buffer.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/allocator.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/allocator.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/backend.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/backend.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/backend_manager.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/backend_manager.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/cache.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/cache.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/executable.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/executable.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/host_tensor.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/host_tensor.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/performance_counter.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/tensor.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/tensor.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/shape.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/shape.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/shape_util.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/shape_util.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/slice_plan.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/slice_plan.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/specialize_function.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/specialize_function.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/state/bernoulli_rng_state.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/state/bernoulli_rng_state.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/state/uniform_rng_state.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/state/uniform_rng_state.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/strides.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/strides.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/type/bfloat16.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/type/bfloat16.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/type/float16.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/type/float16.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/type/element_type.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/type.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/type.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/util.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/util.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/validation_util.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/validation_util.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/variant.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/variant.hpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/dynamic/dynamic_backend.cpp
/home/sasha/workspace/projects/dldt/ngraph/src/ngraph/runtime/dynamic/dynamic_backend.hpp

/home/sasha/workspace/projects/dldt/inference-engine/thirdparty/pugixml/src/pugixml.cpp
/home/sasha/workspace/projects/dldt/inference-engine/thirdparty/pugixml/src/pugiconfig.hpp
/home/sasha/workspace/projects/dldt/inference-engine/thirdparty/pugixml/src/pugixml.hpp

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
