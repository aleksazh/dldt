#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_deconv_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>
#include "ie_parallel.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNDeconvolutionNode::MKLDNNDeconvolutionNode(const InferenceEngine::CNNLayerPtr& layer,
                                                 const mkldnn::engine& eng, int socket) : MKLDNNNode(layer, eng, socket) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:                                                   const mkldnn::engine& eng, int socket) : MKLDNNNode(layer, eng, socket) {" << std::endl;
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
}

void MKLDNNDeconvolutionNode::getSupportedDescriptors() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:  void MKLDNNDeconvolutionNode::getSupportedDescriptors() {" << std::endl;
    if (!descs_fwd.empty() && !descs_bwd.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    auto * deconvLayer = dynamic_cast<DeconvolutionLayer*>(getCnnLayer().get());
    if (deconvLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert deconvolution layer.";
    if (deconvLayer->_weights == nullptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:      if (deconvLayer->_weights == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "Weights are empty for layer: " << deconvLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use ReadWeights and SetWeights methods of InferenceEngine::CNNNetReader"
                           << " to load them from .bin part of the IR";
    }
    withGroups = (deconvLayer->_group > 1);
    isDW = withGroups && deconvLayer->_group == deconvLayer->_out_depth &&
            deconvLayer->_group == deconvLayer->input()->getDims()[1];
    withBiases = (deconvLayer->_biases != nullptr && deconvLayer->_biases->size() != 0);
    if (withBiases)
        biases = deconvLayer->_biases;

    /* Original layout format for deconv weights is iohw (from Caffe).
     * We specify oihw, but mean iohw, because there are no more
     * suitable format in MKLDNN.
     */
    SizeVector weightDims;
    if (withGroups) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:      if (withGroups) {" << std::endl;
        weightDims = {
                deconvLayer->_group,
                deconvLayer->input()->getTensorDesc().getDims()[1] / deconvLayer->_group,
                deconvLayer->_out_depth / deconvLayer->_group,
        };
        groupNum = deconvLayer->_group;
    } else {
        weightDims = {
                deconvLayer->input()->getTensorDesc().getDims()[1],
                deconvLayer->_out_depth
        };
    }
    for (int i = 1; i <= deconvLayer->_kernel.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:      for (int i = 1; i <= deconvLayer->_kernel.size(); i++) {" << std::endl;
        weightDims.push_back(deconvLayer->_kernel[deconvLayer->_kernel.size() - i]);
    }

    internalBlobs.push_back(createInternalBlob(weightDims, true));

    invertVectorCopyUtoI(deconvLayer->_stride, stride);
    for (int i = 1; i <= deconvLayer->_dilation.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:      for (int i = 1; i <= deconvLayer->_dilation.size(); i++) {" << std::endl;
        dilation.push_back(static_cast<int>(deconvLayer->_dilation[deconvLayer->_dilation.size() - i]) - 1);
    }
    auto allPads = getPaddings(*deconvLayer);
    invertVectorCopyUtoI(allPads.begin, paddingL);
    invertVectorCopyUtoI(allPads.end, paddingR);

    weightsDims = MKLDNNDims(weightDims);

    for (int i = 0; i < paddingR.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:      for (int i = 0; i < paddingR.size(); i++) {" << std::endl;
        int with_group = (withGroups) ? 1 : 0;
        int krn = weightsDims[with_group + 2 + i];
        int src = getChildEdgeAt(0)->getDims()[2 + i];
        int dst = getParentEdgeAt(0)->getDims()[2 + i];

        krn = (krn - 1)*(dilation[i] + 1) + 1;
        int calc_dst = (src - krn + paddingL[i]) / stride[i] + 1;
        paddingR[i] = (dst - calc_dst) * stride[i];
    }

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:      for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {" << std::endl;
        MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, format);
        MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, format);
        createDescriptor({in_candidate}, {out_candidate});
    }
}

void MKLDNNDeconvolutionNode::execute(mkldnn::stream strm) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:  void MKLDNNDeconvolutionNode::execute(mkldnn::stream strm) {" << std::endl;
    if (prim) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:      if (prim) {" << std::endl;
        strm.submit({*prim});
    }
    if (withBiases) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:      if (withBiases) {" << std::endl;
        const auto *bias = biases->buffer().as<const float*>();
        auto biasSize = biases->size();

        auto dst = getChildEdgeAt(0)->getBlob();

        float *output = dst->buffer().as<float *>() + dst->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto dims_size = dst->getTensorDesc().getDims().size();
        auto layout = dst->getTensorDesc().getLayout();

        const size_t N = dst->getTensorDesc().getDims()[0];
        size_t C = dst->getTensorDesc().getBlockingDesc().getBlockDims()[1] / groupNum;
        if (C < 1) C = 1;
        const size_t D = dims_size > 4 ? dst->getTensorDesc().getDims()[dims_size - 3] : 1lu;
        const size_t H = dst->getTensorDesc().getDims()[dims_size - 2];
        const size_t W = dst->getTensorDesc().getDims()[dims_size - 1];
        size_t blkC = 1lu;
        if (layout == BLOCKED && dst->getTensorDesc().getBlockingDesc().getBlockDims().size() > 5) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:          if (layout == BLOCKED && dst->getTensorDesc().getBlockingDesc().getBlockDims().size() > 5) {" << std::endl;
            blkC = dst->getTensorDesc().getBlockingDesc().getBlockDims().size() > 5 ?
                   dst->getTensorDesc().getBlockingDesc().getBlockDims()[5] :
                   1lu;
        } else if (layout == BLOCKED && dst->getTensorDesc().getBlockingDesc().getBlockDims().size() > 4) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:          } else if (layout == BLOCKED && dst->getTensorDesc().getBlockingDesc().getBlockDims().size() > 4) {" << std::endl;
            blkC = dst->getTensorDesc().getBlockingDesc().getBlockDims()[4];
        }

        auto strides = dst->getTensorDesc().getBlockingDesc().getStrides();
        int output_size = strides[0] * N - dst->getTensorDesc().getBlockingDesc().getOffsetPadding();

        parallel_for5d(N, C, D, H, W, [&](size_t n, size_t c, size_t d, size_t h, size_t w) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:          parallel_for5d(N, C, D, H, W, [&](size_t n, size_t c, size_t d, size_t h, size_t w) {" << std::endl;
            for (size_t g = 0; g < groupNum; g++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:              for (size_t g = 0; g < groupNum; g++) {" << std::endl;
                const size_t off = n * strides[0]
                                 + (g * C + c) * strides[1]
                                 + d * strides[dims_size - 3]
                                 + h * strides[dims_size - 2]
                                 + w * strides[dims_size - 1];
                if (off >= output_size) continue;
                auto o = &output[off];
                int gcb = g * C * blkC + c * blkC;
                for (int bc = 0; bc < blkC; ++bc) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:                  for (int bc = 0; bc < blkC; ++bc) {" << std::endl;
                    int index = gcb + bc;
                    if (index < biasSize)
                        o[bc] += bias[index];
                }
            }
        });
    }
}

bool MKLDNNDeconvolutionNode::created() const {
    return getType() == Deconvolution;
}

void MKLDNNDeconvolutionNode::createPrimitive() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:  void MKLDNNDeconvolutionNode::createPrimitive() {" << std::endl;
    if (prim)
        return;

    auto prim_desc = createPrimitiveDescriptor<convolution_backward_data::primitive_desc,
            convolution_backward_data::desc, convolution_forward::primitive_desc>();

    prim.reset(new convolution_backward_data(prim_desc,
            getParentEdgeAt(0)->getMemory().GetPrimitive(),
            internalBlobMemory[0]->GetPrimitive(),
            getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

void MKLDNNDeconvolutionNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                               const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:                                                 const std::vector<InferenceEngine::TensorDesc> &outputDesc) {" << std::endl;
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(outputDesc[0]);
    auto in_fmt = in_candidate.getFormat();
    auto out_fmt = out_candidate.getFormat();
    int O_IND = withGroups ? 1 : 0;
    int I_IND = withGroups ? 2 : 1;

    // grouping and autoblicking is not compatible
    if ((withGroups && !isDW) && (in_candidate.blocksExtended() || out_candidate.blocksExtended()))
        return;

    MKLDNNMemoryDesc wgh_candidate{weightsDims, in_candidate.getDataType(), memory::any};
    for (auto alg : {algorithm::convolution_winograd, algorithm::convolution_direct}) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:      for (auto alg : {algorithm::convolution_winograd, algorithm::convolution_direct}) {" << std::endl;
        try {
            std::shared_ptr<mkldnn::convolution_forward::desc> conv_desc;
            conv_desc.reset(new convolution_forward::desc(prop_kind::forward_inference, alg,
                                                          out_candidate, wgh_candidate, in_candidate, stride, dilation,
                                                          paddingL, paddingR, padding_kind::zero));

            std::shared_ptr<mkldnn::convolution_backward_data::desc> deconv_desc;
            deconv_desc.reset(new convolution_backward_data::desc(alg, out_candidate, wgh_candidate,
                                                        in_candidate, stride, dilation, paddingL, paddingR,
                                                        padding_kind::zero));
            descs_fwd.push_back(conv_desc);
            descs_bwd.push_back(deconv_desc);

            descs.emplace_back(deconv_desc,
                               std::shared_ptr<convolution_forward::primitive_desc>(
                                       new convolution_forward::primitive_desc(*conv_desc, getEngine())));
        } catch(...) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:          } catch(...) {" << std::endl;}
    }
}

MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:  MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {" << std::endl;
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.diff_dst_primitive_desc(idx).desc());
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_deconv_node.cpp:  MKLDNNMemoryDesc MKLDNNDeconvolutionNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {" << std::endl;
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.diff_src_primitive_desc(idx).desc());
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}
REG_MKLDNN_PRIM_FOR(MKLDNNDeconvolutionNode, Deconvolution);
