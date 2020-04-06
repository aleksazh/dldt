#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_node.h"
#include "mkldnn_extension_mngr.h"

#include "details/caseless.hpp"
#include <vector>
#include <string>
#include <limits>
#include <cstdint>
#include <unordered_map>

#include <nodes/mkldnn_batchnorm_node.h>
#include <nodes/mkldnn_concat_node.h>
#include <nodes/mkldnn_conv_node.h>
#include <nodes/mkldnn_crop_node.h>
#include <nodes/mkldnn_deconv_node.h>
#include <nodes/mkldnn_eltwise_node.h>
#include <nodes/mkldnn_gemm_node.h>
#include <nodes/mkldnn_fullyconnected_node.h>
#include <nodes/mkldnn_generic_node.h>
#include <nodes/mkldnn_input_node.h>
#include <nodes/mkldnn_lrn_node.h>
#include <nodes/mkldnn_pooling_node.h>
#include <nodes/mkldnn_power_node.h>
#include <nodes/mkldnn_activation_node.h>
#include <nodes/mkldnn_reorder_node.h>
#include <nodes/mkldnn_reshape_node.h>
#include <nodes/mkldnn_roi_pooling_node.h>
#include <nodes/mkldnn_depthwise_node.h>
#include <nodes/mkldnn_softmax_node.h>
#include <nodes/mkldnn_tile_node.h>
#include <nodes/mkldnn_split_node.h>
#include <nodes/mkldnn_permute_node.h>
#include <nodes/mkldnn_memory_node.hpp>
#include <nodes/mkldnn_rnn.h>
#include <nodes/mkldnn_quantize_node.h>
#include <nodes/mkldnn_bin_conv_node.h>
#include <nodes/mkldnn_def_conv_node.h>
#include <nodes/mkldnn_mvn_node.h>
#include <nodes/mkldnn_tensoriterator_node.h>
#include <mkldnn_types.h>
#include "mkldnn_extension_utils.h"
#include "mkldnn_plugin.h"
#include "ie_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;

using namespace InferenceEngine::details;

namespace MKLDNNPlugin {
static const InferenceEngine::details::caseless_unordered_map<std::string, Type> type_to_name_tbl = {
        { "Unknown", Unknown },
        { "Input", Input },
        { "Const", Input },
        { "Output", Output },
        { "Reorder", Reorder },
        { "Convolution", Convolution },
        { "ReLU", Activation },
        { "ELU", Activation },
        { "Sigmoid", Activation },
        { "Logistic", Activation },
        { "TanH", Activation },
        { "ReLU6", Activation },
        { "Exp", Activation },
        { "Not", Activation },
        { "Activation", Activation },
        { "ScaleShift", Depthwise },
        { "PReLU", Depthwise },
        { "Clamp", Activation },
        { "Norm", Lrn },
        { "LRN", Lrn },
        { "Pooling", Pooling },
        { "FullyConnected", FullyConnected },
        { "InnerProduct", FullyConnected },
        { "Gemm", Gemm },
        { "Softmax", SoftMax },
        { "SoftMax", SoftMax },
        { "Split", Split },
        { "Slice", Split },
        { "Concat", Concatenation },
        { "Power", Power },
        { "Deconvolution", Deconvolution },
        { "Eltwise", Eltwise },
        { "Crop", Crop },
        { "Reshape", Reshape },
        { "Tile", Tile },
        { "SimplerNMS", SimplerNMS },
        { "ROIPooling", ROIPooling },
        { "BatchNormalization", BatchNormalization },
        { "Flatten", Flatten },
        { "Permute", Permute },
        { "Copy", Copy },
        { "LSTMCell", RNNCell },
        { "GRUCell", RNNCell },
        { "RNNCell", RNNCell },
        { "LSTMSequence", RNNSeq },
        { "GRUSequence", RNNSeq },
        { "RNNSequence", RNNSeq },
        { "Quantize", Quantize },
        { "FakeQuantize", Quantize },
        { "BinaryConvolution", BinaryConvolution },
        { "DeformableConvolution", DeformableConvolution },
        { "TensorIterator", TensorIterator },
        { "MemoryInput", MemoryInput},  // for construction from name ctor, arbitrary name is used
        { "Memory", MemoryOutput },  // for construction from layer ctor
        { "Convert", Convert },
        { "MVN", MVN},
};

Type TypeFromName(const std::string type) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  Type TypeFromName(const std::string type) {" << std::endl;
    auto itType = type_to_name_tbl.find(type);
    if (type_to_name_tbl.end() != itType) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (type_to_name_tbl.end() != itType) {" << std::endl;
        return itType->second;
    } else {
        return Unknown;
    }
}

}  //  namespace MKLDNNPlugin

std::shared_ptr<MKLDNNNodesHolder> MKLDNNNode::GetNodesHolder() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  std::shared_ptr<MKLDNNNodesHolder> MKLDNNNode::GetNodesHolder() {" << std::endl;
    static std::shared_ptr<MKLDNNNodesHolder> localHolder;
    if (localHolder == nullptr) {
        std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (localHolder == nullptr) {" << std::endl;
        localHolder = std::make_shared<MKLDNNNodesHolder>();
    }
    return localHolder;
}

void MKLDNNNode::AddNode(const std::string& name, CreatorByLayerFunction factory) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::AddNode(const std::string& name, CreatorByLayerFunction factory) {" << std::endl;
    GetNodesHolder()->nodes[name] = factory;
}

MKLDNNNode::MKLDNNNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int _socket)
        : cnnLayer(layer), name(layer->name), typeStr(layer->type), type(TypeFromName(layer->type)), engine(eng),
          selectedPrimitiveDescriptorIndex(-1), permanent(false), temporary(false), constant(ConstantType::Unknown),
          profilingTask(name), socket(_socket) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:            profilingTask(name), socket(_socket) {" << std::endl;
    if (!layer->outData.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (!layer->outData.empty()) {" << std::endl;
        for (const auto& outData : layer->outData) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (const auto& outData : layer->outData) {" << std::endl;
            outDims.emplace_back(outData->getDims());
        }
    } else {
        if (!(CaselessEq<std::string>()(layer->type, "memory") ||
            CaselessEq<std::string>()(layer->type, "memoryinput") ||
            CaselessEq<std::string>()(layer->type, "output") ||
            CaselessEq<std::string>()(layer->type, "reorder"))) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              CaselessEq<std::string>()(layer->type, 'reorder'))) {" << std::endl;
            THROW_IE_EXCEPTION << "Inappropriate layer type: " << layer->type << " name: " << layer->name;
        }
    }

    for (const auto& inData : layer->insData) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (const auto& inData : layer->insData) {" << std::endl;
        inDims.emplace_back(inData.lock()->getDims());
    }
    if (layer->params.find("PrimitivesPriority") != layer->params.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (layer->params.find('PrimitivesPriority') != layer->params.end()) {" << std::endl;
        std::istringstream stream(layer->params["PrimitivesPriority"]);
        std::string str;
        while (getline(stream, str, ',')) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          while (getline(stream, str, ',')) {" << std::endl;
            if (str.substr(0, 4) != "cpu:")
                continue;
            implPriorities.push_back(parse_impl_name(str));
            if (implPriorities[implPriorities.size() - 1] == impl_desc_type::unknown &&
                    str != "cpu:unknown")
                THROW_IE_EXCEPTION << "Unsupported CPU implementation " << str << " for node " << getName();
        }
    }
}

void MKLDNNNode::addEdge(const MKLDNNEdgeWeakPtr& edge) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::addEdge(const MKLDNNEdgeWeakPtr& edge) {" << std::endl;
    auto edgePtr = edge.lock();
    if (!edgePtr)
        return;
    auto parentPtr = edgePtr->getParent();
    auto childPtr = edgePtr->getChild();
    if (!parentPtr || !childPtr)
        return;

    parentPtr->childEdges.push_back(edge);
    childPtr->parentEdges.push_back(edge);
}

void MKLDNNNode::removeEdge(const MKLDNNEdgeWeakPtr& edge) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::removeEdge(const MKLDNNEdgeWeakPtr& edge) {" << std::endl;
    auto edgePtr = edge.lock();
    if (!edgePtr)
        return;
    auto parentPtr = edgePtr->getParent();
    auto childPtr = edgePtr->getChild();
    if (!parentPtr || !childPtr)
        return;
    for (auto it = childPtr->parentEdges.begin(); it != childPtr->parentEdges.end(); it++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto it = childPtr->parentEdges.begin(); it != childPtr->parentEdges.end(); it++) {" << std::endl;
        auto parentEdge = (*it).lock();
        if (parentEdge && parentEdge->getChild() == childPtr && parentEdge->getParent() == parentPtr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (parentEdge && parentEdge->getChild() == childPtr && parentEdge->getParent() == parentPtr) {" << std::endl;
            childPtr->parentEdges.erase(it);
            break;
        }
    }
    for (auto it = parentPtr->childEdges.begin(); it != parentPtr->childEdges.end(); it++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto it = parentPtr->childEdges.begin(); it != parentPtr->childEdges.end(); it++) {" << std::endl;
        auto childEdge = (*it).lock();
        if (childEdge && childEdge->getChild() == childPtr && childEdge->getParent() == parentPtr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (childEdge && childEdge->getChild() == childPtr && childEdge->getParent() == parentPtr) {" << std::endl;
            parentPtr->childEdges.erase(it);
            break;
        }
    }
}

void MKLDNNNode::remove() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::remove() {" << std::endl;
    auto parent_edges = parentEdges;
    for (const auto &parentEdge : parent_edges) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (const auto &parentEdge : parent_edges) {" << std::endl;
        removeEdge(parentEdge);
    }
    auto child_edges = childEdges;
    for (const auto &childEdge : child_edges) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (const auto &childEdge : child_edges) {" << std::endl;
        removeEdge(childEdge);
    }
}

MKLDNNNode* MKLDNNNode::CreateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng,
                                   const MKLDNNExtensionManager::Ptr& extMgr, int socket) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                                     const MKLDNNExtensionManager::Ptr& extMgr, int socket) {" << std::endl;
    MKLDNNNode *newNode = nullptr;
    auto nodesHolder = GetNodesHolder();

    if (nodesHolder->nodes.find("Generic") != nodesHolder->nodes.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (nodesHolder->nodes.find('Generic') != nodesHolder->nodes.end()) {" << std::endl;
        std::unique_ptr<MKLDNNNode> ol(nodesHolder->nodes["Generic"](layer, eng, socket));
        if (ol != nullptr && ol->created(extMgr))
            newNode = ol.release();
    }
    if (newNode == nullptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (newNode == nullptr) {" << std::endl;
        for (auto maker : nodesHolder->nodes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (auto maker : nodesHolder->nodes) {" << std::endl;
            std::unique_ptr<MKLDNNNode> ol(maker.second(layer, eng, socket));
            if (ol != nullptr && ol->created(extMgr)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              if (ol != nullptr && ol->created(extMgr)) {" << std::endl;
                newNode = ol.release();
                break;
            }
        }
    }

    //  WA-start : TI node requires all attributes to construct internal subgpath
    //             including extManager, socket and mkldnn::eng.
#if defined (COMPILED_CPU_MKLDNN_TENSORITERATOR_NODE)
    MKLDNNTensorIteratorNode *ti = dynamic_cast<MKLDNNTensorIteratorNode*>(newNode);
    if (ti != nullptr)
        ti->setExtManager(extMgr);
#endif
    //  WA-end

    if (!newNode)
        THROW_IE_EXCEPTION << "Unsupported primitive of type: " << layer->type << " name: " << layer->name;

    return newNode;
}

bool MKLDNNNode::isEdgesEmpty(const std::vector<MKLDNNEdgeWeakPtr>& edges) const {
    for (auto &edge : edges) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto &edge : edges) {" << std::endl;
        if (edge.lock())
            return false;
    }
    return true;
}

void MKLDNNNode::selectOptimalPrimitiveDescriptor() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::selectOptimalPrimitiveDescriptor() {" << std::endl;
    selectPreferPrimitiveDescriptor(getPrimitivesPriority());
}

void MKLDNNNode::selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority) {" << std::endl;
    for (auto& type : priority) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto& type : priority) {" << std::endl;
        int selectedPrimitive = -1;
        int equalsFormatCount = -1;
        for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {" << std::endl;
            impl_desc_type supportedType = getSupportedPrimitiveDescriptors()[i].getImplementationType();
            if (type == supportedType) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              if (type == supportedType) {" << std::endl;
                int equalsLocalFormatCount = 0;
                if (getSupportedPrimitiveDescriptors()[i].getConfig().inConfs.size() > getParentEdges().size())
                    continue;
                for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[i].getConfig().inConfs.size(); j++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                  for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[i].getConfig().inConfs.size(); j++) {" << std::endl;
                    auto parentEdge = getParentEdgeAt(j);
                    auto parentPtr = parentEdge->getParent();
                    auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

                    if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                      if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {" << std::endl;
                        int inNum = parentEdge->getInputNum();
                        if (inNum < 0 || inNum >= parent_spd->getConfig().outConfs.size()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                          if (inNum < 0 || inNum >= parent_spd->getConfig().outConfs.size()) {" << std::endl;
                            inNum = 0;
                        }
                        if (MKLDNNExtensionUtils::initTensorsAreEqual(
                                getSupportedPrimitiveDescriptors()[i].getConfig().inConfs[j].desc,
                                parent_spd->getConfig().outConfs[inNum].desc)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                                  parent_spd->getConfig().outConfs[inNum].desc)) {" << std::endl;
                            equalsLocalFormatCount++;
                        }
                    }
                }
                if (equalsLocalFormatCount > equalsFormatCount) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                  if (equalsLocalFormatCount > equalsFormatCount) {" << std::endl;
                    equalsFormatCount = equalsLocalFormatCount;
                    selectedPrimitive = static_cast<int>(i);
                }
            }
        }
        if (selectedPrimitive >= 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (selectedPrimitive >= 0) {" << std::endl;
            selectPrimitiveDescriptorByIndex(selectedPrimitive);
            return;
        }
    }

    if (getSupportedPrimitiveDescriptors().empty())
        THROW_IE_EXCEPTION << "Supported primitive descriptors list is empty for node: " << getName();
    // fallback. If there are no primitives from priority list just select a first
    selectPrimitiveDescriptorByIndex(0);
}

bool MKLDNNNode::canBeInPlace() const {
    if (getParentEdges().size() != 1 || getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1 ||
            (getParentEdgeAt(0)->getParent()->isConstant() && !getParentEdgeAt(0)->getChild()->isConstant()))
        return false;

    // TODO: we need to extend this logic to properly handle all possible inplace conflicts
    if (getParentEdges().size() == 1 && getParentEdgeAt(0)->getParent()->getType() == Reshape) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (getParentEdges().size() == 1 && getParentEdgeAt(0)->getParent()->getType() == Reshape) {" << std::endl;
        auto reshapeNode = getParentEdgeAt(0)->getParent();
        if (reshapeNode->getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1)
            return false;
    }

    MKLDNNDims dims = getParentEdgeAt(0)->getDims();
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {" << std::endl;
        if (getChildEdgeAt(cIdx)->getDims() != dims) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (getChildEdgeAt(cIdx)->getDims() != dims) {" << std::endl;
            return false;
        }
    }
    return true;
}

void MKLDNNNode::resolveNotAllocatedEdges() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::resolveNotAllocatedEdges() {" << std::endl;
    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (!selected_pd)
        THROW_IE_EXCEPTION << "Cannot find selected primitive descriptor for node: " << getName();
    for (size_t i = 0; i < getParentEdges().size() && i < selected_pd->getConfig().inConfs.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (size_t i = 0; i < getParentEdges().size() && i < selected_pd->getConfig().inConfs.size(); i++) {" << std::endl;
        auto parentEdge = getParentEdgeAt(i);

        if (parentEdge->getStatus() != MKLDNNEdge::Status::NotAllocated || selected_pd->getConfig().inConfs[i].inPlace < 0)
            continue;

        auto * memPtr = reinterpret_cast<char*>(parentEdge->getMemory().GetData());
        parentEdge->getMemoryPtr().reset(new MKLDNNMemory(getEngine()));
        parentEdge->getMemoryPtr()->Create(MKLDNNMemoryDesc(selected_pd->getConfig().inConfs[i].desc), memPtr);

        parentEdge->changeStatus(MKLDNNEdge::Status::Allocated);
    }
    for (size_t i = 0; i < getChildEdges().size() && i < selected_pd->getConfig().outConfs.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (size_t i = 0; i < getChildEdges().size() && i < selected_pd->getConfig().outConfs.size(); i++) {" << std::endl;
        auto childEdge = getChildEdgeAt(i);

        if (childEdge->getStatus() != MKLDNNEdge::Status::NotAllocated || selected_pd->getConfig().outConfs[i].inPlace < 0)
            continue;

        auto * memPtr = reinterpret_cast<char*>(childEdge->getMemory().GetData());
        childEdge->getMemoryPtr().reset(new MKLDNNMemory(getEngine()));
        childEdge->getMemoryPtr()->Create(MKLDNNMemoryDesc(selected_pd->getConfig().outConfs[i].desc), memPtr);

        childEdge->changeStatus(MKLDNNEdge::Status::Allocated);
    }
}

std::string MKLDNNNode::getPrimitiveDescriptorType() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  std::string MKLDNNNode::getPrimitiveDescriptorType() {" << std::endl;
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();

    impl_desc_type type = impl_desc_type::undef;
    if (selectedPrimitiveDesc) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (selectedPrimitiveDesc) {" << std::endl;
        type = selectedPrimitiveDesc->getImplementationType();
    }

    std::string str_type;

    auto add_type = [&](std::string t) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      auto add_type = [&](std::string t) {" << std::endl;
        if (!str_type.empty() && t.c_str()[0] != '_')
            str_type += "_";
        str_type += t;
    };

#define SEARCH_TYPE(_type)                                          \
    if ((type & impl_desc_type::_type) == impl_desc_type::_type)    \
        add_type(#_type)

    SEARCH_TYPE(undef);
    SEARCH_TYPE(reorder);
    SEARCH_TYPE(jit);
    SEARCH_TYPE(gemm);
    SEARCH_TYPE(ref);

    SEARCH_TYPE(avx512);
    SEARCH_TYPE(avx2);
    SEARCH_TYPE(avx);
    SEARCH_TYPE(sse42);
    SEARCH_TYPE(blas);
    SEARCH_TYPE(any);
    SEARCH_TYPE(uni);

    SEARCH_TYPE(winograd);
    SEARCH_TYPE(_dw);
    SEARCH_TYPE(_1x1);

    if (type == impl_desc_type::unknown)
        str_type = "unknown";
    else if (str_type.empty())
        str_type = "undef";

    // adding layer precision to the performance counters as one of the token
    // currently we treat a layer executing in int8 mode if its input is I8 or U8. if input is U8, we still
    // add I8 since I8 is special placeholder. The real calc precision might be quite complex and in most cases
    // it is mixed precision.
    if (selectedPrimitiveDesc) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (selectedPrimitiveDesc) {" << std::endl;
        if (!selectedPrimitiveDesc->getConfig().inConfs.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (!selectedPrimitiveDesc->getConfig().inConfs.empty()) {" << std::endl;
            if (selectedPrimitiveDesc->getConfig().inConfs[0].desc.getPrecision() != InferenceEngine::Precision::U8) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              if (selectedPrimitiveDesc->getConfig().inConfs[0].desc.getPrecision() != InferenceEngine::Precision::U8) {" << std::endl;
                str_type += "_" + std::string(selectedPrimitiveDesc->getConfig().inConfs[0].desc.getPrecision().name());
            } else {
                str_type += "_I8";
            }
        }
    }

    return str_type;
}

const MKLDNNEdgePtr MKLDNNNode::getParentEdgeAt(size_t idx) const {
    if (idx >= parentEdges.size())
        THROW_IE_EXCEPTION << "Node " << getName() << " contains less parent edges than " << idx;
    auto parentEdgePtr = parentEdges[idx].lock();
    if (!parentEdgePtr)
        THROW_IE_EXCEPTION << "Node " << getName() << " contains empty parent edge for index " << idx;
    return parentEdgePtr;
}

const MKLDNNEdgePtr MKLDNNNode::getChildEdgeAt(size_t idx) const {
    if (idx >= childEdges.size())
        THROW_IE_EXCEPTION << "Node " << getName() << " contains less child edges than " << idx;
    auto childEdgePtr = childEdges[idx].lock();
    if (!childEdgePtr)
        THROW_IE_EXCEPTION << "Node " << getName() << " contains empty child edge for index " << idx;
    return childEdgePtr;
}

const std::vector<MKLDNNEdgePtr> MKLDNNNode::getParentEdgesAtPort(size_t idx) const {
    if (idx >= inDims.size())
        THROW_IE_EXCEPTION << "Node " << getName() << " contains less input ports than " << idx;

    std::vector<MKLDNNEdgePtr> res;
    for (auto &edge_w : parentEdges) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto &edge_w : parentEdges) {" << std::endl;
        auto edge = edge_w.lock();
        if (!edge)
            THROW_IE_EXCEPTION << "Node " << getName() << " contains dead weak ptr";
        if (edge->getOutputNum() == idx) res.push_back(edge);
    }
    return res;
}

const std::vector<MKLDNNEdgePtr> MKLDNNNode::getChildEdgesAtPort(size_t idx) const {
    if (idx >= outDims.size())
        THROW_IE_EXCEPTION << "Node " << getName() << " contains less output ports than " << idx;

    std::vector<MKLDNNEdgePtr> res;
    for (auto &edge_w : childEdges) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto &edge_w : childEdges) {" << std::endl;
        auto edge = edge_w.lock();
        if (!edge)
            THROW_IE_EXCEPTION << "Node " << getName() << " contains dead weak ptr";
        if (edge->getInputNum() == idx) res.push_back(edge);
    }
    return res;
}


std::vector<memory::format> MKLDNNNode::getAvailableFormatsForDims(const MKLDNNDims &dims) const {
    if (dims.ndims() == 0)
        return {memory::format::x};
    else if (dims.ndims() == 1)
        return {memory::format::x};
    else if (dims.ndims() == 2)
        return {memory::format::nc};
    else if (dims.ndims() == 3)
        return {memory::format::tnc, memory::format::ntc};
    else if (dims.ndims() == 4)
        return {memory::format::nchw, memory::format::nChw8c, memory::format::nChw16c};
    else if (dims.ndims() == 5)
        return {memory::format::ncdhw, memory::format::nCdhw8c, memory::format::nCdhw16c};
    return {memory::format::any};
}

void MKLDNNNode::execute(mkldnn::stream strm) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::execute(mkldnn::stream strm) {" << std::endl;
    if (prim) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (prim) {" << std::endl;
        strm.submit({*prim});
    }
}

void MKLDNNNode::initSupportedPrimitiveDescriptors() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::initSupportedPrimitiveDescriptors() {" << std::endl;
    if (!supportedPrimitiveDescriptors.empty())
        return;

    for (auto& desc : descs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto& desc : descs) {" << std::endl;
        auto itpd = desc.createPrimitiveDescriptorIterator(engine);
        while (itpd.is_not_end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          while (itpd.is_not_end()) {" << std::endl;
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              for (size_t i = 0; i < descInputNumbers(desc); i++) {" << std::endl;
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getSrcMemDesc(itpd, i));
                config.inConfs.push_back(dataConfig);
            }

            std::vector<mkldnn::memory::format> outFormats;
            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              for (size_t i = 0; i < descOutputNumbers(desc); i++) {" << std::endl;
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getDstMemDesc(itpd, i));
                config.outConfs.push_back(dataConfig);

                auto primDesc = itpd.fetch();
                auto dstPrimDesc = mkldnn_primitive_desc_query_pd(primDesc.get(), mkldnn::convert_to_c(dst_pd), 0);
                if (dstPrimDesc) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                  if (dstPrimDesc) {" << std::endl;
                    outFormats.emplace_back(static_cast<memory::format>(itpd.dst_primitive_desc().desc().data.format));
                } else {
                    // This path is needed to correctly handle Deconvolution node
                    auto diffSrcPrimDesc = mkldnn_primitive_desc_query_pd(primDesc.get(), mkldnn::convert_to_c(diff_src_pd), 0);
                    if (diffSrcPrimDesc) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                      if (diffSrcPrimDesc) {" << std::endl;
                        outFormats.emplace_back(static_cast<memory::format>(itpd.diff_src_primitive_desc().desc().data.format));
                    }
                }
            }
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type, outFormats);
            itpd++;
        }
    }
}

void MKLDNNNode::initDescriptor(const InferenceEngine::LayerConfig &config) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::initDescriptor(const InferenceEngine::LayerConfig &config) {" << std::endl;
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (!selectedPD) {" << std::endl;
        return;
    }
    std::vector<InferenceEngine::TensorDesc> inDescs;
    for (const auto& inConf : config.inConfs)
        inDescs.push_back(inConf.desc);
    std::vector<InferenceEngine::TensorDesc> outDescs;
    for (const auto& outConf : config.outConfs)
        outDescs.push_back(outConf.desc);
    createDescriptor({inDescs}, {outDescs});

    std::shared_ptr<mkldnn::primitive_attr> attr = initPrimitiveAttr();

    InferenceEngine::LayerConfig rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;
    for (size_t j = 0; j < descs.size(); j++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (size_t j = 0; j < descs.size(); j++) {" << std::endl;
        const auto &desc = descs[j];
        std::shared_ptr<primitive_desc_iterator> itpd;
        if (attr == nullptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (attr == nullptr) {" << std::endl;
            itpd = std::make_shared<primitive_desc_iterator>(desc.createPrimitiveDescriptorIterator(engine));
        } else {
            itpd = std::make_shared<primitive_desc_iterator>(desc.createPrimitiveDescriptorIterator(engine, *(attr.get())));
        }
        while (itpd->is_not_end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          while (itpd->is_not_end()) {" << std::endl;
            InferenceEngine::LayerConfig cfg;
            cfg.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              for (size_t i = 0; i < descInputNumbers(desc); i++) {" << std::endl;
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(*itpd, i);
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              for (size_t i = 0; i < descOutputNumbers(desc); i++) {" << std::endl;
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(*itpd, i);
                cfg.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd->get_impl_info_str().c_str());
            if (selected_count == selectedPrimitiveDescriptorIndex) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              if (selected_count == selectedPrimitiveDescriptorIndex) {" << std::endl;
                if (impl_type != selectedPD->getImplementationType()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                  if (impl_type != selectedPD->getImplementationType()) {" << std::endl;
                    THROW_IE_EXCEPTION << "Cannot get the original layer configuration!";
                }
                rightConfig = cfg;
            }
            if (j == descs.size() - 1) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              if (j == descs.size() - 1) {" << std::endl;
                if (impl_type == selectedPD->getImplementationType()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:                  if (impl_type == selectedPD->getImplementationType()) {" << std::endl;
                    rightConfig = config;
                }
            }
            selected_count++;
            (*itpd)++;
        }
    }

    if (descs.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (descs.empty()) {" << std::endl;
        const auto& selectedConfig = selectedPD->getConfig();
        if (selectedConfig.inConfs.size() != config.inConfs.size() || selectedConfig.outConfs.size() != config.outConfs.size())
            return;

        for (size_t i = 0; i < selectedConfig.inConfs.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (size_t i = 0; i < selectedConfig.inConfs.size(); i++) {" << std::endl;
            if (selectedConfig.inConfs[i].desc.getLayout() != InferenceEngine::Layout::ANY &&
                !MKLDNNExtensionUtils::initTensorsAreEqual(selectedConfig.inConfs[i].desc, config.inConfs[i].desc))
                THROW_IE_EXCEPTION << "Incorrect descriptor for node: " << getName();
        }

        for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {" << std::endl;
            if (selectedConfig.outConfs[i].desc.getLayout() != InferenceEngine::Layout::ANY &&
                !MKLDNNExtensionUtils::initTensorsAreEqual(selectedConfig.outConfs[i].desc, config.outConfs[i].desc))
                THROW_IE_EXCEPTION << "Incorrect descriptor for node: " << getName();
        }
        rightConfig = config;
    }

    selectedPD->getConfig() = rightConfig;
}

InferenceEngine::Blob::Ptr MKLDNNNode::createInternalBlob(InferenceEngine::SizeVector dims, bool weights, bool isGrouped) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  InferenceEngine::Blob::Ptr MKLDNNNode::createInternalBlob(InferenceEngine::SizeVector dims, bool weights, bool isGrouped) {" << std::endl;
    auto checkSize = [](size_t dst_size, size_t src_size) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      auto checkSize = [](size_t dst_size, size_t src_size) {" << std::endl;
        if (dst_size < src_size) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (dst_size < src_size) {" << std::endl;
            THROW_IE_EXCEPTION << "Cannot create internal buffer. Buffer can be overrun.";
        }
    };
    auto * wLayer = dynamic_cast<InferenceEngine::WeightableLayer*>(getCnnLayer().get());
    if (wLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get weightable layer for node " << getName() << ".";

    InferenceEngine::Blob::Ptr blb = weights ? wLayer->_weights : wLayer->_biases;

    if (blb == nullptr)
        THROW_IE_EXCEPTION << "Cannot get internal blob layer for node " << getName() << ".";

    auto intLayout = getWeightsLayoutByDims(dims, isGrouped);

    InferenceEngine::TensorDesc desc(blb->getTensorDesc().getPrecision(), dims, intLayout);

    auto fillInternalBlob = [&](char *data, size_t intBuffSize) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      auto fillInternalBlob = [&](char *data, size_t intBuffSize) {" << std::endl;
        size_t offset = blb->byteSize();
        checkSize(intBuffSize, offset);
        ie_memcpy(data, intBuffSize, blb->buffer(), blb->byteSize());
        data += blb->byteSize();
        for (const auto &merged : getMergeWith()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (const auto &merged : getMergeWith()) {" << std::endl;
            wLayer = dynamic_cast<InferenceEngine::WeightableLayer*>(merged->getCnnLayer().get());
            if (wLayer == nullptr)
                THROW_IE_EXCEPTION << "Cannot convert merged weightable layer for node "
                                   << getName() << ".";
            blb = weights ? wLayer->_weights : wLayer->_biases;

            if (blb == nullptr)
                THROW_IE_EXCEPTION << "Cannot get internal blob layer for node " << getName() << ".";
            offset += blb->byteSize();
            checkSize(intBuffSize, offset);
            ie_memcpy(data, intBuffSize, blb->buffer(), blb->byteSize());
            data += blb->byteSize();
        }
    };

    Blob::Ptr internalBlob;
    if (blb->getTensorDesc().getPrecision() == Precision::BIN) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (blb->getTensorDesc().getPrecision() == Precision::BIN) {" << std::endl;
        internalBlob = InferenceEngine::make_shared_blob<int8_t>(desc);
    } else if (blb->getTensorDesc().getPrecision() == Precision::I8) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      } else if (blb->getTensorDesc().getPrecision() == Precision::I8) {" << std::endl;
        internalBlob = InferenceEngine::make_shared_blob<int8_t>(desc);
    } else if (blb->getTensorDesc().getPrecision() == Precision::I32) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      } else if (blb->getTensorDesc().getPrecision() == Precision::I32) {" << std::endl;
        internalBlob = InferenceEngine::make_shared_blob<int32_t>(desc);
    } else {
        internalBlob = InferenceEngine::make_shared_blob<float>(desc);
    }
    internalBlob->allocate();
    char *data = internalBlob->buffer();
    size_t intBuffSize = internalBlob->byteSize();

    fillInternalBlob(data, intBuffSize);

    return internalBlob;
}

void MKLDNNNode::prepareMemory(const PrimitiveDescInfo *selected_pd, mkldnn::primitive_desc_iterator& itpd) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::prepareMemory(const PrimitiveDescInfo *selected_pd, mkldnn::primitive_desc_iterator& itpd) {" << std::endl;
    for (size_t i = 0; i < getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (size_t i = 0; i < getChildEdges().size(); i++) {" << std::endl;
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (size_t i = 0; i < getParentEdges().size(); i++) {" << std::endl;
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }
    std::vector<MKLDNNMemoryDesc> intDescs;
    for (auto &it : internalBlobDesc)
        intDescs.push_back(it(itpd, 0));

    internalBlobMemory.clear();
    for (size_t i = 0; i < internalBlobs.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (size_t i = 0; i < internalBlobs.size(); i++) {" << std::endl;
        const auto &internalBlob = internalBlobs[i];

        auto create = [&] () {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          auto create = [&] () {" << std::endl;
            auto newDesc = MKLDNNMemoryDesc(internalBlob->getTensorDesc());
            auto newFormat = newDesc.getFormat();
            if (newFormat == mkldnn::memory::ncdhw) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              if (newFormat == mkldnn::memory::ncdhw) {" << std::endl;
                newFormat = mkldnn::memory::goihw;
            }
            if (newFormat == mkldnn::memory::nchw) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              if (newFormat == mkldnn::memory::nchw) {" << std::endl;
                newFormat = mkldnn::memory::oihw;
            }

            MKLDNNMemory memory{ engine };
            memory.Create(MKLDNNMemoryDesc(newDesc.getDims(), newDesc.getDataType(), newFormat), internalBlob->buffer());

            MKLDNNMemoryPtr _ptr = MKLDNNMemoryPtr(new MKLDNNMemory(engine));
            _ptr->Create(intDescs[i]);
            _ptr->SetData(memory);

            return _ptr;
        };

        MKLDNNMemoryPtr ptr;
        if (weight_caching) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (weight_caching) {" << std::endl;
            const uint64_t data_hash = Engine::GetWeightsSharing(socket)->GetHashFunc().hash(
                    internalBlob->buffer(), internalBlob->byteSize());

            const std::string string_hash = name + "_" + std::to_string(i)
                                            + "_" + std::to_string(internalBlob->byteSize())
                                            + "_" + std::to_string(data_hash);

            ptr = Engine::GetWeightsSharing(socket)->findOrCreate(string_hash, create);
        } else {
            ptr = create();
        }

        internalBlobMemory.push_back(ptr);
    }
}

bool MKLDNNNode::isInplace() const {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();

    for (auto &in : config.inConfs) if (in.inPlace >= 0) return true;
    for (auto &out : config.outConfs) if (out.inPlace >= 0) return true;
    return false;
}

bool MKLDNNNode::isConstant() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  bool MKLDNNNode::isConstant() {" << std::endl;
    if (constant == ConstantType::Unknown) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (constant == ConstantType::Unknown) {" << std::endl;
        std::vector<MKLDNNNodePtr> checkNodes;
        for (size_t i = 0; i < getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (size_t i = 0; i < getChildEdges().size(); i++) {" << std::endl;
            checkNodes.push_back(getChildEdgeAt(i)->getChild());
        }
        while (constant != ConstantType::NoConst && !checkNodes.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          while (constant != ConstantType::NoConst && !checkNodes.empty()) {" << std::endl;
            constant = checkNodes.front()->checkConstant(LOOK_DOWN, checkNodes);
            checkNodes.erase(checkNodes.begin());
        }
        if (constant != ConstantType::Const) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (constant != ConstantType::Const) {" << std::endl;
            constant = ConstantType::Unknown;
            checkNodes.clear();
            for (size_t i = 0; i < getParentEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              for (size_t i = 0; i < getParentEdges().size(); i++) {" << std::endl;
                checkNodes.push_back(getParentEdgeAt(i)->getParent());
            }
            while (constant != ConstantType::NoConst && !checkNodes.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              while (constant != ConstantType::NoConst && !checkNodes.empty()) {" << std::endl;
                constant = checkNodes.front()->checkConstant(LOOK_UP, checkNodes);
                checkNodes.erase(checkNodes.begin());
            }
        }
        if (constant == ConstantType::Unknown)
            constant = ConstantType::NoConst;
    }
    return constant == ConstantType::Const;
}

MKLDNNNode::ConstantType MKLDNNNode::checkConstant(LOOK look, std::vector<MKLDNNNodePtr>& checkNodes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  MKLDNNNode::ConstantType MKLDNNNode::checkConstant(LOOK look, std::vector<MKLDNNNodePtr>& checkNodes) {" << std::endl;
    if (constant == ConstantType::Unknown) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (constant == ConstantType::Unknown) {" << std::endl;
        if (look == LOOK_DOWN) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          if (look == LOOK_DOWN) {" << std::endl;
            for (size_t i = 0; i < getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              for (size_t i = 0; i < getChildEdges().size(); i++) {" << std::endl;
                if (std::find(checkNodes.begin(), checkNodes.end(), getChildEdgeAt(i)->getChild()) == checkNodes.end())
                    checkNodes.push_back(getChildEdgeAt(i)->getChild());
            }
        } else {
            for (size_t i = 0; i < getParentEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              for (size_t i = 0; i < getParentEdges().size(); i++) {" << std::endl;
                if (std::find(checkNodes.begin(), checkNodes.end(), getParentEdgeAt(i)->getParent()) == checkNodes.end())
                    checkNodes.push_back(getParentEdgeAt(i)->getParent());
            }
        }
    }
    return constant;
}

void MKLDNNNode::addOriginalLayer(const InferenceEngine::CNNLayerPtr &layer) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::addOriginalLayer(const InferenceEngine::CNNLayerPtr &layer) {" << std::endl;
    if (!layer) return;
    if (originalLayers.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (originalLayers.empty()) {" << std::endl;
        originalLayers = layer->name;
    } else {
        originalLayers += "," + layer->name;
    }
}

void MKLDNNNode::cleanup() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::cleanup() {" << std::endl;
    internalBlobs.clear();
    cnnLayer.reset();

    for (auto it : fusedWith) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto it : fusedWith) {" << std::endl;
        it->cleanup();
    }

    for (auto it : mergedWith) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto it : mergedWith) {" << std::endl;
        it->cleanup();
    }
}

const std::vector<impl_desc_type>& MKLDNNNode::getPrimitivesPriority() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  const std::vector<impl_desc_type>& MKLDNNNode::getPrimitivesPriority() {" << std::endl;
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::jit_uni_dw,
            impl_desc_type::jit_uni_1x1,
            impl_desc_type::jit_uni,
            impl_desc_type::jit_avx512_dw,
            impl_desc_type::jit_avx512_1x1,
            impl_desc_type::jit_avx512,
            impl_desc_type::jit_avx2_dw,
            impl_desc_type::jit_avx2_1x1,
            impl_desc_type::jit_avx2,
            impl_desc_type::jit_avx_dw,
            impl_desc_type::jit_avx_1x1,
            impl_desc_type::jit_avx,
            impl_desc_type::jit_sse42_dw,
            impl_desc_type::jit_sse42_1x1,
            impl_desc_type::jit_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_avx,
            impl_desc_type::gemm_sse42,
            impl_desc_type::ref_any,
            impl_desc_type::ref,
    };
    for (const auto& impl : priorities) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (const auto& impl : priorities) {" << std::endl;
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

bool MKLDNNNode::isUninitTensorDesc(const InferenceEngine::TensorDesc& desc) const {
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return true;

    if (desc.getBlockingDesc().getOffsetPadding() == std::numeric_limits<size_t>::max())
        return true;

    for (size_t i = 0; i < desc.getBlockingDesc().getOrder().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (size_t i = 0; i < desc.getBlockingDesc().getOrder().size(); i++) {" << std::endl;
        if (desc.getBlockingDesc().getOffsetPaddingToData()[i] == std::numeric_limits<size_t>::max() ||
                desc.getBlockingDesc().getStrides()[i] == std::numeric_limits<size_t>::max())
            return true;
    }

    return false;
}

InferenceEngine::TensorDesc MKLDNNNode::getConfiguredInputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const {
    if (!isUninitTensorDesc(config.inConfs[idx].desc))
        return config.inConfs[idx].desc;

    int num = getParentEdgeAt(idx)->getInputNum();
    auto *selectedPD = getParentEdgeAt(idx)->getParent()->getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        THROW_IE_EXCEPTION << "Cannot get selected primitive descriptor for node: " << getParentEdgeAt(idx)->getParent()->getName();

    if (selectedPD->getConfig().outConfs.size() <= num)
        num = 0;

    if (config.inConfs[idx].inPlace >= 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (config.inConfs[idx].inPlace >= 0) {" << std::endl;
        return getConfiguredOutputDesc(config, static_cast<size_t>(config.inConfs[idx].inPlace));
    }

    if (num >= 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (num >= 0) {" << std::endl;
        auto parentConf = selectedPD->getConfig().outConfs[num];
        parentConf.desc.setPrecision(config.inConfs[idx].desc.getPrecision());
        if (isUninitTensorDesc(parentConf.desc) && parentConf.inPlace >= 0)
            getParentEdgeAt(idx)->getParent()->initOptimalPrimitiveDescriptor();
        parentConf = getParentEdgeAt(idx)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num];
        if (!isUninitTensorDesc(parentConf.desc) &&
            MKLDNNExtensionUtils::initTensorsAreEqual(parentConf.desc, config.inConfs[idx].desc)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              MKLDNNExtensionUtils::initTensorsAreEqual(parentConf.desc, config.inConfs[idx].desc)) {" << std::endl;
            return parentConf.desc;
        }

        if (config.inConfs[idx].desc.getLayout() == InferenceEngine::Layout::ANY &&
            parentConf.desc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              parentConf.desc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
            return InferenceEngine::TensorDesc(parentConf.desc.getPrecision(),
                                               parentConf.desc.getDims(), {
                                                       parentConf.desc.getBlockingDesc().getBlockDims(),
                                                       parentConf.desc.getBlockingDesc().getOrder()
                                               });
        }
    }

    if (config.inConfs[idx].desc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (config.inConfs[idx].desc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
        return InferenceEngine::TensorDesc(config.inConfs[idx].desc.getPrecision(),
                                           config.inConfs[idx].desc.getDims(), {
                                                   config.inConfs[idx].desc.getBlockingDesc().getBlockDims(),
                                                   config.inConfs[idx].desc.getBlockingDesc().getOrder()
                                           });
    }

    return InferenceEngine::TensorDesc(config.inConfs[idx].desc.getPrecision(),
                                       config.inConfs[idx].desc.getDims(),
                                       InferenceEngine::TensorDesc::getLayoutByDims(config.inConfs[idx].desc.getDims()));
}

InferenceEngine::TensorDesc MKLDNNNode::getConfiguredOutputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const {
    if (!isUninitTensorDesc(config.outConfs[idx].desc))
        return config.outConfs[idx].desc;

    int num = getChildEdgeAt(idx)->getOutputNum();
    auto *selectedPD = getChildEdgeAt(idx)->getChild()->getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        THROW_IE_EXCEPTION << "Cannot get selected primitive descriptor for node: " << getChildEdgeAt(idx)->getChild()->getName();

    if (selectedPD->getConfig().inConfs.size() <= num)
        num = 0;

    if (config.outConfs[idx].inPlace >= 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (config.outConfs[idx].inPlace >= 0) {" << std::endl;
        return getConfiguredInputDesc(config, static_cast<size_t>(config.outConfs[idx].inPlace));
    }

    if (num >= 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (num >= 0) {" << std::endl;
        auto childConf = selectedPD->getConfig().inConfs[num];
        childConf.desc.setPrecision(config.outConfs[idx].desc.getPrecision());
        if (isUninitTensorDesc(childConf.desc) && childConf.inPlace >= 0)
            getChildEdgeAt(idx)->getChild()->initOptimalPrimitiveDescriptor();
        childConf = getChildEdgeAt(idx)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
        if (!isUninitTensorDesc(childConf.desc) &&
            MKLDNNExtensionUtils::initTensorsAreEqual(childConf.desc, config.outConfs[idx].desc)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              MKLDNNExtensionUtils::initTensorsAreEqual(childConf.desc, config.outConfs[idx].desc)) {" << std::endl;
            return childConf.desc;
        }
        if (config.outConfs[idx].desc.getLayout() == InferenceEngine::Layout::ANY &&
            childConf.desc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:              childConf.desc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
            return InferenceEngine::TensorDesc(childConf.desc.getPrecision(),
                                               childConf.desc.getDims(), {
                                                       childConf.desc.getBlockingDesc().getBlockDims(),
                                                       childConf.desc.getBlockingDesc().getOrder()
                                               });
        }
    }

    if (config.outConfs[idx].desc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (config.outConfs[idx].desc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
        return InferenceEngine::TensorDesc(config.outConfs[idx].desc.getPrecision(),
                                                                config.outConfs[idx].desc.getDims(), {
                                                                        config.outConfs[idx].desc.getBlockingDesc().getBlockDims(),
                                                                        config.outConfs[idx].desc.getBlockingDesc().getOrder()
                                                                });
    }

    return InferenceEngine::TensorDesc(config.outConfs[idx].desc.getPrecision(),
                                       config.outConfs[idx].desc.getDims(),
                                       InferenceEngine::TensorDesc::getLayoutByDims(config.outConfs[idx].desc.getDims()));
}

void MKLDNNNode::initOptimalPrimitiveDescriptor() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::initOptimalPrimitiveDescriptor() {" << std::endl;
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (!isInitConfig(config)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (!isInitConfig(config)) {" << std::endl;
        for (size_t i = 0; i < config.inConfs.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (size_t i = 0; i < config.inConfs.size(); i++) {" << std::endl;
            config.inConfs[i].desc = getConfiguredInputDesc(config, i);
        }

        for (size_t i = 0; i < config.outConfs.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (size_t i = 0; i < config.outConfs.size(); i++) {" << std::endl;
            config.outConfs[i].desc = getConfiguredOutputDesc(config, i);
        }
        initDescriptor(config);
    } else if (getType() != RNNSeq && getType() != RNNCell) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      } else if (getType() != RNNSeq && getType() != RNNCell) {" << std::endl;
        initDescriptor(config);
    }
}

bool MKLDNNNode::isInitConfig(const InferenceEngine::LayerConfig& config) const {
    for (const auto& configs : {config.inConfs, config.outConfs}) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (const auto& configs : {config.inConfs, config.outConfs}) {" << std::endl;
        for (const auto &dc : configs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:          for (const auto &dc : configs) {" << std::endl;
            if (isUninitTensorDesc(dc.desc))
                return false;
        }
    }
    return true;
}

MKLDNNMemoryDesc MKLDNNNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  MKLDNNMemoryDesc MKLDNNNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {" << std::endl;
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.src_primitive_desc(idx).desc());
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

MKLDNNMemoryDesc MKLDNNNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  MKLDNNMemoryDesc MKLDNNNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {" << std::endl;
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.dst_primitive_desc(idx).desc());
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

int MKLDNNNode::batchToProcess() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  int MKLDNNNode::batchToProcess() {" << std::endl;
    return dynBatchLim == 0 ? getMaxBatch() : std::min<int>(getMaxBatch(), dynBatchLim);
}

int MKLDNNNode::getMaxBatch() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  int MKLDNNNode::getMaxBatch() {" << std::endl;
    // FIXME: batch != 0 dims number
    if (!inDims.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (!inDims.empty()) {" << std::endl;
        if (inDims[0].ndims())
            return inDims[0][0];
        else
            return 1;
    }
    if (!outDims.empty() && outDims[0].ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (!outDims.empty() && outDims[0].ndims()) {" << std::endl;
        if (outDims[0].ndims())
            return outDims[0][0];
        else
            return 1;
    }
    return 0;
}

void MKLDNNNode::setDynamicBatchLim(int lim) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  void MKLDNNNode::setDynamicBatchLim(int lim) {" << std::endl;
    dynBatchLim = lim;
    if (prim) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      if (prim) {" << std::endl;
        prim.setBatchLimit(batchToProcess(), getParentEdges().size(), getChildEdges().size());
    }
}

bool MKLDNNNode::isFusedWith(Type fusedNodeType) const {
    for (auto fusedNode : fusedWith) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      for (auto fusedNode : fusedWith) {" << std::endl;
        if (fusedNode->type == fusedNodeType)
            return true;
    }

    return false;
}

Layout MKLDNNNode::getWeightsLayoutByDims(SizeVector dims, bool isGrouped) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:  Layout MKLDNNNode::getWeightsLayoutByDims(SizeVector dims, bool isGrouped) {" << std::endl;
    switch (dims.size()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_node.cpp:      switch (dims.size()) {" << std::endl;
        case 0:
            return Layout::SCALAR;
        case 1:
            return Layout::C;
        case 2:
            return Layout::NC;
        case 3:
            return Layout::CHW;
        case 4:
            return Layout::OIHW;
        case 5:
            return isGrouped ? Layout::GOIHW : Layout::OIDHW;
        case 6:
            return isGrouped ? Layout::GOIDHW : Layout::BLOCKED;
        default:
            return Layout::BLOCKED;
    }
}
