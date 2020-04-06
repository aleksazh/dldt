#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_softmax_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNSoftMaxNode::MKLDNNSoftMaxNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket) :
        MKLDNNNode(layer, eng, socket) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:          MKLDNNNode(layer, eng, socket) {" << std::endl;}

void MKLDNNSoftMaxNode::getSupportedDescriptors() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:  void MKLDNNSoftMaxNode::getSupportedDescriptors() {" << std::endl;
    if (descs.size())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    SoftMaxLayer* smLayer = dynamic_cast<SoftMaxLayer*>(getCnnLayer().get());
    if (smLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert softmax layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    axis = smLayer->axis;

    if (axis >= getParentEdgeAt(0)->getDims().ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:      if (axis >= getParentEdgeAt(0)->getDims().ndims()) {" << std::endl;
        THROW_IE_EXCEPTION << "Incorrect axis!";
    }

    if (getParentEdgeAt(0)->getDims().ndims() == 3) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:      if (getParentEdgeAt(0)->getDims().ndims() == 3) {" << std::endl;
        MKLDNNMemoryDesc in_candidate(getParentEdgeAt(0)->getDims(), inputDataType, memory::format::blocked);
        createDescriptor({in_candidate}, {});
    }

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:      for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {" << std::endl;
        MKLDNNDims dims = getParentEdgeAt(0)->getDims();
        if (MKLDNNMemoryDesc(dims, inputDataType, format).blocksExtended())
            continue;

        MKLDNNMemoryDesc in_candidate(dims, inputDataType, format);

        createDescriptor({in_candidate}, {});
    }
}

void MKLDNNSoftMaxNode::createPrimitive() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:  void MKLDNNSoftMaxNode::createPrimitive() {" << std::endl;
    if (prim)
        return;

    memory::desc in_candidate = getParentEdgeAt(0)->getMemory().GetDescriptor();
    MKLDNNDescriptor desc(std::shared_ptr<softmax_forward::desc>(
            new softmax_forward::desc(prop_kind::forward_scoring, in_candidate, axis)));
    descs[0] = desc;
    std::shared_ptr<softmax_forward::desc> selected_desc_ptr = descs[0];

    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set for node " << getName() << ".";

    auto prim_desc = softmax_forward::primitive_desc(*selected_desc_ptr, getEngine());
    primitive_desc_iterator itpd = descs[0].createPrimitiveDescriptorIterator(getEngine());

    while (itpd.is_not_end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:      while (itpd.is_not_end()) {" << std::endl;
        impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());
        auto primitiveDescriptor = getSelectedPrimitiveDescriptor();
        if ((primitiveDescriptor != nullptr) && (impl_type == primitiveDescriptor->getImplementationType())) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:          if ((primitiveDescriptor != nullptr) && (impl_type == primitiveDescriptor->getImplementationType())) {" << std::endl;
            itpd.getPrimitiveDescriptor(prim_desc);
            break;
        }
        itpd++;
    }

    prim.reset(new softmax_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

bool MKLDNNSoftMaxNode::created() const {
    return getType() == SoftMax;
}

void MKLDNNSoftMaxNode::initOptimalPrimitiveDescriptor() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:  void MKLDNNSoftMaxNode::initOptimalPrimitiveDescriptor() {" << std::endl;
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    if (config.inConfs.size() != 1 || config.outConfs.size() != 1 ||
            (!isUninitTensorDesc(config.inConfs[0].desc) &&
                    !isUninitTensorDesc(config.outConfs[0].desc) && config.inConfs[0].desc != config.outConfs[0].desc))
        THROW_IE_EXCEPTION << "Layer " << getName() << " has incorrect selected config!";

    if (!isUninitTensorDesc(config.inConfs[0].desc)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:      if (!isUninitTensorDesc(config.inConfs[0].desc)) {" << std::endl;
        config.outConfs[0].desc = config.inConfs[0].desc;
    } else if (!isUninitTensorDesc(config.outConfs[0].desc)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:      } else if (!isUninitTensorDesc(config.outConfs[0].desc)) {" << std::endl;
        config.inConfs[0].desc = config.outConfs[0].desc;
    } else {
        config.outConfs[0].desc = config.inConfs[0].desc = getConfiguredInputDesc(config, 0);
    }

    initDescriptor(config);
}

void MKLDNNSoftMaxNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                         const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_softmax_node.cpp:                                           const std::vector<InferenceEngine::TensorDesc> &outputDesc) {" << std::endl;
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);

    MKLDNNDescriptor desc(std::shared_ptr<softmax_forward::desc>(
            new softmax_forward::desc(prop_kind::forward_scoring, in_candidate, axis)));
    descs.push_back(desc);
}
REG_MKLDNN_PRIM_FOR(MKLDNNSoftMaxNode, Softmax);
