#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_split_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <map>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <limits>
#include <ie_parallel.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNSplitNode::MKLDNNSplitNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket) :
        MKLDNNNode(layer, eng, socket) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          MKLDNNNode(layer, eng, socket) {" << std::endl;}

void MKLDNNSplitNode::getSupportedDescriptors() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:  void MKLDNNSplitNode::getSupportedDescriptors() {" << std::endl;
    auto * splitLayer = dynamic_cast<SplitLayer*>(getCnnLayer().get());

    if (splitLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert split layer.";

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input nodes.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output nodes.";

    axis = splitLayer->_axis;
    if (axis >= getParentEdgeAt(0)->getDims().ndims())
        THROW_IE_EXCEPTION << "Invalid value of axis parameter in split layer";
}

void MKLDNNSplitNode::initSupportedPrimitiveDescriptors() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:  void MKLDNNSplitNode::initSupportedPrimitiveDescriptors() {" << std::endl;
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.inConfs[0].desc = MKLDNNMemoryDesc(srcDims, inputDataType, memory::format::any);
    config.outConfs.resize(outDims.size());

    std::vector<memory::format> outFormats;

    auto axis_size = 0;
    auto dstFirstDims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < outDims.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < outDims.size(); i++) {" << std::endl;
        auto o_Dims = outDims[i];
        if (dstFirstDims.ndims() != o_Dims.ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          if (dstFirstDims.ndims() != o_Dims.ndims()) {" << std::endl;
            THROW_IE_EXCEPTION << "Split " << getName() << " supports only output blob with equal number of dimensions";
        }

        config.outConfs[i].inPlace = -1;
        config.outConfs[i].constant = false;
        config.outConfs[i].desc = MKLDNNMemoryDesc(o_Dims, outputDataType, memory::format::any);
        outFormats.push_back(memory::format::any);

        axis_size += o_Dims[axis];
        for (size_t j = 0; j < dstFirstDims.ndims(); j++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (size_t j = 0; j < dstFirstDims.ndims(); j++) {" << std::endl;
            if (j == axis)
                continue;
            if (o_Dims[j] != dstFirstDims[j])
                THROW_IE_EXCEPTION << "Split " << getName() << " has incorrect output dimensions";
        }
    }
    dstFirstDims[axis] = axis_size;
    if (dstFirstDims.size() != srcDims.size())
        THROW_IE_EXCEPTION << "The sizes of input blob and sum of output blobs are not equal.";
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, outFormats);

    auto numOfDim = static_cast<size_t>(srcDims.ndims());

    SizeVector order;
    SizeVector offsets(numOfDim, 0lu);
    size_t offset = (std::numeric_limits<size_t>::max)();
    for (size_t i = 0; i < numOfDim; i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < numOfDim; i++) {" << std::endl;
        order.push_back(i);
    }

    SizeVector strides(numOfDim);
    strides[numOfDim - 1] = 1;
    for (size_t i = 2; i <= numOfDim; i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 2; i <= numOfDim; i++) {" << std::endl;
        if (numOfDim - i < axis) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          if (numOfDim - i < axis) {" << std::endl;
            strides[numOfDim - i] = (std::numeric_limits<size_t>::max)();
        } else {
            strides[numOfDim - i] = strides[numOfDim - i + 1] * srcDims[numOfDim - i + 1];
        }
    }

    config.inConfs[0].desc = TensorDesc(Precision::FP32, srcDims.ToSizeVector(), {srcDims.ToSizeVector(), order, offset, offsets, strides});
    outFormats.clear();
    for (size_t i = 0; i < outDims.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < outDims.size(); i++) {" << std::endl;
        auto dims = outDims[i].ToSizeVector();
        config.outConfs[i].inPlace = 0;
        config.outConfs[i].desc = TensorDesc(Precision::FP32, dims,
                                            {dims, order, offset, offsets, strides});
        outFormats.push_back(MKLDNNMemory::Convert(config.outConfs[i].desc.getLayout()));
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFormats);

    if ((numOfDim != 4 && numOfDim != 5) || axis != 1)
        return;

    order.push_back(1);
    numOfDim = order.size();
    offsets = SizeVector(numOfDim, 0lu);

    // nChw8c and nChw16c
    for (size_t sizeS : {8lu, 16lu}) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t sizeS : {8lu, 16lu}) {" << std::endl;
        SizeVector blkDims = srcDims.ToSizeVector();
        if (blkDims[1] % sizeS)
            continue;
        blkDims[1] = blkDims[1] / sizeS + (blkDims[1] % sizeS ? 1lu : 0lu);
        blkDims.push_back(sizeS);

        strides.resize(numOfDim);
        strides[numOfDim - 1] = 1lu;
        for (size_t i = 2; i <= numOfDim; i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (size_t i = 2; i <= numOfDim; i++) {" << std::endl;
            if (numOfDim - i < axis) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              if (numOfDim - i < axis) {" << std::endl;
                strides[numOfDim - i] = (std::numeric_limits<size_t>::max)();
            } else {
                strides[numOfDim - i] = strides[numOfDim - i + 1] * blkDims[numOfDim - i + 1];
            }
        }
        config.inConfs[0].desc = TensorDesc(Precision::FP32, srcDims.ToSizeVector(), {blkDims, order, offset, offsets, strides});

        outFormats.clear();
        bool canInplace = true;
        for (size_t i = 0; i < outDims.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (size_t i = 0; i < outDims.size(); i++) {" << std::endl;
            auto dims = outDims[i].ToSizeVector();
            blkDims = dims;

            if (blkDims[1] % sizeS) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              if (blkDims[1] % sizeS) {" << std::endl;
                canInplace = false;
                break;
            }
            blkDims[1] = blkDims[1] / sizeS + (blkDims[1] % sizeS ? 1lu : 0lu);
            blkDims.push_back(sizeS);
            config.outConfs[i].desc = TensorDesc(Precision::FP32, dims, {blkDims, order, offset, offsets, strides});

            outFormats.emplace_back(MKLDNNMemory::Convert(config.outConfs[i].desc.getLayout()));
        }
        if (canInplace)
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFormats);
    }
}

void MKLDNNSplitNode::createPrimitive() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:  void MKLDNNSplitNode::createPrimitive() {" << std::endl;
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    for (size_t i = 0; i < getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < getChildEdges().size(); i++) {" << std::endl;
        if (!getChildEdgeAt(i)->getMemoryPtr() || !getChildEdgeAt(i)->getMemory().GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    }
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    canUseOptimizedImpl = true;
    if (axis != 1)
        canUseOptimizedImpl = false;

    if (getParentEdgeAt(0)->getBlob()->getTensorDesc().getLayout() != NHWC &&
        getParentEdgeAt(0)->getBlob()->getTensorDesc().getLayout() != NDHWC)
        canUseOptimizedImpl = false;

    for (size_t i = 0; i < getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < getChildEdges().size(); i++) {" << std::endl;
        if (getChildEdgeAt(i)->getBlob()->getTensorDesc().getLayout() != NCHW &&
            getChildEdgeAt(i)->getBlob()->getTensorDesc().getLayout() != NCDHW)
            canUseOptimizedImpl = false;
    }
}

void MKLDNNSplitNode::optimizedImpl(size_t MB) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:  void MKLDNNSplitNode::optimizedImpl(size_t MB) {" << std::endl;
    const int ndims = getParentEdgeAt(0)->getDims().ndims();
    const size_t IC = getParentEdgeAt(0)->getDims()[1];
    const size_t D = ndims == 5 ? getParentEdgeAt(0)->getDims()[ndims - 3] : 1;
    const size_t H = getParentEdgeAt(0)->getDims()[ndims - 2];
    const size_t W = getParentEdgeAt(0)->getDims()[ndims - 1];

    auto srcBlob = getParentEdgeAt(0)->getBlob();
    const auto *srcData = srcBlob->cbuffer().as<const float *>();
    for (size_t i = 0, sIdx = 0; i < getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0, sIdx = 0; i < getChildEdges().size(); i++) {" << std::endl;
        auto dstBlob = getChildEdgeAt(i)->getBlob();
        auto *dstData = dstBlob->buffer().as<float *>();

        const size_t OC = getChildEdgeAt(i)->getDims()[1];

        size_t innerSize = 1;
        for (size_t j = axis; j < dstBlob->getTensorDesc().getDims().size(); j++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (size_t j = axis; j < dstBlob->getTensorDesc().getDims().size(); j++) {" << std::endl;
            innerSize *= dstBlob->getTensorDesc().getDims()[j];
        }

        auto srcPtr = srcData + srcBlob->getTensorDesc().offset(sIdx);

        parallel_for4d(MB, D, H, W, [&](size_t b, size_t d, size_t h, size_t w) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          parallel_for4d(MB, D, H, W, [&](size_t b, size_t d, size_t h, size_t w) {" << std::endl;
            for (size_t c = 0; c < OC; c++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              for (size_t c = 0; c < OC; c++) {" << std::endl;
                size_t srcOff = b*D*H*W*IC + d*H*W*IC + h*W*IC + w*IC + c;
                size_t dstOff = b*OC*D*H*W + c*D*H*W + d*H*W + h*W + w;

                dstData[dstOff] = srcPtr[srcOff];
            }
        });

        sIdx += innerSize;
    }
}

void MKLDNNSplitNode::execute(mkldnn::stream strm) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:  void MKLDNNSplitNode::execute(mkldnn::stream strm) {" << std::endl;
    if (isOptimized())
        return;

    // FIXME: add more optimal implementation
    MKLDNNDims par_dims = getParentEdgeAt(0)->getDims();
    int MB = batchToProcess();
    auto srcBlob = getParentEdgeAt(0)->getBlob();
    const auto *srcData = srcBlob->cbuffer().as<const float *>();

    size_t outerSize = 1;
    for (int i = 0; i < axis; i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (int i = 0; i < axis; i++) {" << std::endl;
        if (i == 0)
            outerSize *= MB;
        else
            outerSize *= srcBlob->getTensorDesc().getDims()[i];
    }

    if (canUseOptimizedImpl) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      if (canUseOptimizedImpl) {" << std::endl;
        optimizedImpl(MB);
        return;
    }

    size_t srcSize = getParentEdgeAt(0)->getMemory().GetSize();
    size_t src_batch_off = srcBlob->getTensorDesc().offset(srcBlob->size() / outerSize)
                           - srcBlob->getTensorDesc().offset(0);

    for (size_t i = 0, sIdx = 0; i < getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0, sIdx = 0; i < getChildEdges().size(); i++) {" << std::endl;
        auto dstBlob = getChildEdgeAt(i)->getBlob();
        auto *dstData = dstBlob->buffer().as<float *>();

        size_t innerSize = 1;
        for (size_t j = axis; j < dstBlob->getTensorDesc().getDims().size(); j++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (size_t j = axis; j < dstBlob->getTensorDesc().getDims().size(); j++) {" << std::endl;
            innerSize *= dstBlob->getTensorDesc().getDims()[j];
        }

        size_t dst_batch_off = dstBlob->getTensorDesc().offset(innerSize) - dstBlob->getTensorDesc().offset(0);

        for (size_t dIdx = 0; dIdx < innerSize; dIdx++, sIdx++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (size_t dIdx = 0; dIdx < innerSize; dIdx++, sIdx++) {" << std::endl;
            for (unsigned b = 0; b < outerSize; b++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              for (unsigned b = 0; b < outerSize; b++) {" << std::endl;
                if (sIdx + b*src_batch_off >= srcSize)
                    THROW_IE_EXCEPTION << "Incorrect configuration of split layer " << getName() << "!";
                dstData[b * dst_batch_off + dstBlob->getTensorDesc().offset(dIdx)] =
                        srcData[b * src_batch_off + srcBlob->getTensorDesc().offset(sIdx)];
            }
        }
    }
}

bool MKLDNNSplitNode::created() const {
    return getType() == Split;
}

void MKLDNNSplitNode::selectOptimalPrimitiveDescriptor() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:  void MKLDNNSplitNode::selectOptimalPrimitiveDescriptor() {" << std::endl;
    if (implPriorities.size() > 0 && implPriorities[0] == impl_desc_type::ref) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      if (implPriorities.size() > 0 && implPriorities[0] == impl_desc_type::ref) {" << std::endl;
        selectPrimitiveDescriptorByIndex(0);
        return;
    }
    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    bool hasUnknown = false;
    std::vector<size_t> canSelectPrimitive;
    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {" << std::endl;
        bool hasAny = true;
        auto &primDescInfo = supportedPrimitiveDescriptors[i];
        if (primDescInfo.getImplementationType() != impl_desc_type::unknown ||
            primDescInfo.getConfig().outConfs[0].inPlace < 0)
            continue;
        hasUnknown = true;
        for (auto iInfo : primDescInfo.getConfig().inConfs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (auto iInfo : primDescInfo.getConfig().inConfs) {" << std::endl;
            if (iInfo.desc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              if (iInfo.desc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
                hasAny = false;
                break;
            }
        }

        if (hasAny) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          if (hasAny) {" << std::endl;
            for (auto oInfo : primDescInfo.getConfig().outConfs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              for (auto oInfo : primDescInfo.getConfig().outConfs) {" << std::endl;
                if (oInfo.desc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:                  if (oInfo.desc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
                    hasAny = false;
                    break;
                }
            }
        }

        if (!hasAny) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          if (!hasAny) {" << std::endl;
            canSelectPrimitive.push_back(i);
        }
    }

    bool canOptimize = false;
    if (hasUnknown) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      if (hasUnknown) {" << std::endl;
        canOptimize = true;

        if (canSelectPrimitive.size() == 1) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          if (canSelectPrimitive.size() == 1) {" << std::endl;
            selectPrimitiveDescriptorByIndex(static_cast<int>(canSelectPrimitive[0]));
            return;
        }
    }

    std::map<mkldnn::memory::format, size_t> formatFrequency;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < getParentEdges().size(); i++) {" << std::endl;
        auto parentEdge = getParentEdgeAt(i);
        auto parent = parentEdge->getParent();

        if (parent->getSelectedPrimitiveDescriptor() == nullptr)
            continue;

        int outputIndex = parentEdge->getOutputNum();
        if (outputIndex < 0)
            THROW_IE_EXCEPTION << "Cannot find index of output node";
        if (outputIndex >= parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size())
            outputIndex = 0;
        auto outDesc = MKLDNNMemoryDesc(parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIndex].desc);
        if (!outDesc)
            continue;
        if (formatFrequency.find(outDesc.getFormat()) != formatFrequency.end())
            formatFrequency[outDesc.getFormat()] += 1;
        else
            formatFrequency[outDesc.getFormat()] = 1;
    }
    for (size_t i = 0; i < getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < getChildEdges().size(); i++) {" << std::endl;
        auto childEdge = getChildEdgeAt(i);
        auto child = childEdge->getChild();
        if (child->getSelectedPrimitiveDescriptor() == nullptr)
            continue;
        int inputIndex = childEdge->getOutputNum();
        if (inputIndex < 0)
            THROW_IE_EXCEPTION << "Cannot find index of output node";
        if (inputIndex >= child->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size())
            inputIndex = 0;
        auto outDesc = MKLDNNMemoryDesc(child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIndex].desc);
        if (!outDesc)
            continue;
        if (formatFrequency.find(outDesc.getFormat()) != formatFrequency.end())
            formatFrequency[outDesc.getFormat()] += 1;
        else
            formatFrequency[outDesc.getFormat()] = 1;
    }

    size_t maxCount = 0;
    mkldnn::memory::format convertTo = MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims());
    for (auto &it : formatFrequency) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (auto &it : formatFrequency) {" << std::endl;
        if (it.second > maxCount && !MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, it.first).blocksExtended()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          if (it.second > maxCount && !MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, it.first).blocksExtended()) {" << std::endl;
            maxCount = it.second;
            convertTo = it.first;
        }
    }

    // This logic is needed to cover cases when Split node cannot be optimized out for particular block size
    // In general it is significantly better to have additional reorders in graph than to use reference Split implementation
    if (convertTo == memory::nChw16c || convertTo == memory::nCdhw16c ||
        convertTo == memory::nChw8c || convertTo == memory::nCdhw8c) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          convertTo == memory::nChw8c || convertTo == memory::nCdhw8c) {" << std::endl;
        int blockSize = convertTo == memory::nChw16c || convertTo == memory::nCdhw16c ? 16 : 8;
        bool shouldDecreaseBlockSize = false;
        for (auto& parentEdge : getParentEdges()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (auto& parentEdge : getParentEdges()) {" << std::endl;
            if (parentEdge.lock()->getDims()[1] % blockSize != 0)
                shouldDecreaseBlockSize = true;
        }

        for (auto& childEdge : getChildEdges()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (auto& childEdge : getChildEdges()) {" << std::endl;
            if (childEdge.lock()->getDims()[1] % blockSize != 0)
                shouldDecreaseBlockSize = true;
        }

        if (shouldDecreaseBlockSize) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          if (shouldDecreaseBlockSize) {" << std::endl;
            int decreasedBlockSize = 8;
            bool canDecreaseBlockSize = true;
            for (auto &parentEdge : getParentEdges()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              for (auto &parentEdge : getParentEdges()) {" << std::endl;
                if (parentEdge.lock()->getDims()[1] % decreasedBlockSize != 0)
                    canDecreaseBlockSize = false;
            }

            for (auto &childEdge : getChildEdges()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              for (auto &childEdge : getChildEdges()) {" << std::endl;
                if (childEdge.lock()->getDims()[1] % decreasedBlockSize != 0)
                    canDecreaseBlockSize = false;
            }

            if (canDecreaseBlockSize)
                convertTo = getParentEdgeAt(0)->getDims().ndims() == 5 ? memory::nCdhw8c : memory::nChw8c;
            else
                convertTo = MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims());
        }
    }

    if (canOptimize && MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, convertTo).blocksExtended())
        canOptimize = false;
    for (size_t i = 0; canOptimize && i < getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; canOptimize && i < getChildEdges().size(); i++) {" << std::endl;
        if (MKLDNNMemoryDesc(getChildEdgeAt(i)->getDims(), outputDataType, convertTo).blocksExtended())
            canOptimize = false;
    }

    if (canOptimize) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      if (canOptimize) {" << std::endl;
        for (auto supportedPdIndex : canSelectPrimitive) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (auto supportedPdIndex : canSelectPrimitive) {" << std::endl;
            if (MKLDNNMemoryDesc(supportedPrimitiveDescriptors[supportedPdIndex].getConfig().inConfs[0].desc).getFormat() == convertTo) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              if (MKLDNNMemoryDesc(supportedPrimitiveDescriptors[supportedPdIndex].getConfig().inConfs[0].desc).getFormat() == convertTo) {" << std::endl;
                selectPrimitiveDescriptorByIndex(static_cast<int>(supportedPdIndex));
                return;
            }
        }
    }

    for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < supportedPrimitiveDescriptors.size(); i++) {" << std::endl;
        auto &primDescInfo = supportedPrimitiveDescriptors[i];
        if (primDescInfo.getImplementationType() == impl_desc_type::unknown)
            continue;
        if (convertTo == MKLDNNMemoryDesc(supportedPrimitiveDescriptors[i].getConfig().outConfs[0].desc).getFormat()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          if (convertTo == MKLDNNMemoryDesc(supportedPrimitiveDescriptors[i].getConfig().outConfs[0].desc).getFormat()) {" << std::endl;
            size_t num = 0;
            for (num = 0; num < getParentEdges().size(); num++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              for (num = 0; num < getParentEdges().size(); num++) {" << std::endl;
                if (MKLDNNMemoryDesc(getParentEdgeAt(num)->getDims(), inputDataType, convertTo).blocksExtended())
                    break;
            }
            if (num == getParentEdges().size()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              if (num == getParentEdges().size()) {" << std::endl;
                selectPrimitiveDescriptorByIndex(i);
                return;
            }
        }
    }

    selectPrimitiveDescriptorByIndex(0);
}

bool MKLDNNSplitNode::isOptimized() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:  bool MKLDNNSplitNode::isOptimized() {" << std::endl;
    return getSelectedPrimitiveDescriptor() && getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].inPlace >= 0;
}

void MKLDNNSplitNode::initOptimalPrimitiveDescriptor() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:  void MKLDNNSplitNode::initOptimalPrimitiveDescriptor() {" << std::endl;
    if (!isOptimized()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      if (!isOptimized()) {" << std::endl;
        MKLDNNNode::initOptimalPrimitiveDescriptor();
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    for (size_t i = 0; i < config.inConfs.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < config.inConfs.size(); i++) {" << std::endl;
        if (config.inConfs[i].desc.getLayout() == InferenceEngine::Layout::ANY ||
            !isUninitTensorDesc(config.inConfs[i].desc))
            continue;

        int num = getParentEdgeAt(i)->getOutputNum();
        if (getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          if (getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()) {" << std::endl;
            if (num >= 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:              if (num >= 0) {" << std::endl;
                if (isUninitTensorDesc(getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc) &&
                        getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].inPlace >= 0)
                    getParentEdgeAt(i)->getParent()->initOptimalPrimitiveDescriptor();
                if (!isUninitTensorDesc(getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc) &&
                    MKLDNNExtensionUtils::initTensorsAreEqual(
                            getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc,
                            config.inConfs[i].desc)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:                              config.inConfs[i].desc)) {" << std::endl;
                    config.inConfs[i].desc = getParentEdgeAt(i)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num].desc;
                    continue;
                }
            }
        }
        config.inConfs[i].desc = InferenceEngine::TensorDesc(config.inConfs[i].desc.getPrecision(),
                                                              config.inConfs[i].desc.getDims(), {
                                                                      config.inConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.inConfs[i].desc.getBlockingDesc().getOrder()
                                                              });
    }
    const auto& cnnLayer = getCnnLayer();
    if (!cnnLayer)
        THROW_IE_EXCEPTION << "Cannot create Split layer " << getName() << " without CNNLayer!";
    if (config.outConfs.size() != outDims.size())
        THROW_IE_EXCEPTION << "Invalid config for Split layer " << getName();
    size_t offset = 0;
    for (size_t i = 0; i < cnnLayer->outData.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      for (size_t i = 0; i < cnnLayer->outData.size(); i++) {" << std::endl;
        size_t confNum = i;
        config.outConfs[i].desc = InferenceEngine::TensorDesc(config.outConfs[i].desc.getPrecision(),
                                                              config.outConfs[i].desc.getDims(), {
                                                                      config.outConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.outConfs[i].desc.getBlockingDesc().getOrder(),
                                                                      config.inConfs[0].desc.getBlockingDesc().getOffsetPadding() + offset,
                                                                      config.inConfs[0].desc.getBlockingDesc().getOffsetPaddingToData(),
                                                                      config.inConfs[0].desc.getBlockingDesc().getStrides()
                                                              });
        size_t axisSize = 1;
        for (size_t j = axis; j < config.outConfs[confNum].desc.getBlockingDesc().getBlockDims().size(); j++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:          for (size_t j = axis; j < config.outConfs[confNum].desc.getBlockingDesc().getBlockDims().size(); j++) {" << std::endl;
            axisSize *= config.outConfs[confNum].desc.getBlockingDesc().getBlockDims()[j];
        }
        offset += axisSize;
    }
    initDescriptor(config);
}

void MKLDNNSplitNode::setDynamicBatchLim(int lim) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:  void MKLDNNSplitNode::setDynamicBatchLim(int lim) {" << std::endl;
    if (axis == 0)
        THROW_IE_EXCEPTION << "Dynamic batch is not supported by split layer with axis == 0 parameter";

    dynBatchLim = lim;
    if (prim) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/mkldnn_split_node.cpp:      if (prim) {" << std::endl;
        prim.setBatchLimit(batchToProcess(), getParentEdges().size(), getChildEdges().size());
    }
}
REG_MKLDNN_PRIM_FOR(MKLDNNSplitNode, Split);
