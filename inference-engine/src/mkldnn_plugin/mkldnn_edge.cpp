#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_edge.h"
#include "mkldnn_node.h"
#include "mkldnn_extension_utils.h"
#include <blob_factory.hpp>

using namespace mkldnn;
namespace MKLDNNPlugin {

MKLDNNEdge::MKLDNNEdge(const MKLDNNNodePtr &parent, const MKLDNNNodePtr &child, int pr_port, int ch_port) :
        parent(parent), child(child), parent_port(pr_port), child_port(ch_port) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          parent(parent), child(child), parent_port(pr_port), child_port(ch_port) {" << std::endl;}

const MKLDNNNodePtr MKLDNNEdge::getParent() const {
    auto parentPtr = parent.lock();
    if (!parentPtr)
        THROW_IE_EXCEPTION << "Edge contains empty parent node";
    return parentPtr;
}

const MKLDNNNodePtr MKLDNNEdge::getChild() const {
    auto childPtr = child.lock();
    if (!childPtr)
        THROW_IE_EXCEPTION << "Edge contains empty child node";
    return childPtr;
}

bool MKLDNNEdge::isDropped() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  bool MKLDNNEdge::isDropped() {" << std::endl;
    bool not_in_parent = true;
    bool not_in_child = true;

    auto parent_ptr = parent.lock();
    if (parent_ptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (parent_ptr) {" << std::endl;
        for (auto &edge : parent_ptr->childEdges)
            if (edge.lock().get() == this)
                not_in_parent = false;
    }

    auto child_ptr = child.lock();
    if (child_ptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (child_ptr) {" << std::endl;
        for (auto &edge : child_ptr->parentEdges)
            if (edge.lock().get() == this)
                not_in_child = false;
    }
    return not_in_parent && not_in_child;
}

void MKLDNNEdge::drop() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  void MKLDNNEdge::drop() {" << std::endl;
    auto _drop_from = [&] (std::vector<MKLDNNEdgeWeakPtr> &list) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      auto _drop_from = [&] (std::vector<MKLDNNEdgeWeakPtr> &list) {" << std::endl;
        auto myself = std::find_if(list.begin(), list.end(),
                [&] (MKLDNNEdgeWeakPtr edge) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:                  [&] (MKLDNNEdgeWeakPtr edge) {" << std::endl; return edge.lock().get() == this; });

        if (myself != list.end())
            list.erase(myself);
    };

    _drop_from(getParent()->childEdges);
    _drop_from(getChild()->parentEdges);
}


bool MKLDNNEdge::needReorder() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  bool MKLDNNEdge::needReorder() {" << std::endl;
    bool canBeInPlaceConflicts = false;
    auto parentSPD = getParent()->getSelectedPrimitiveDescriptor();
    auto childSPD = getChild()->getSelectedPrimitiveDescriptor();
    if (!parentSPD || !childSPD)
        THROW_IE_EXCEPTION << "Cannot make a decision about reorder. Primitive descriptors weren't selected.";

    int outNumber = getOutputNum();
    int inNumber = getInputNum();
    bool in_place = inPlace();
    bool childCanChangeMem = childSPD->getConfig().outConfs.empty();
    for (const auto conf : childSPD->getConfig().outConfs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (const auto conf : childSPD->getConfig().outConfs) {" << std::endl;
        if (conf.inPlace == outNumber && outNumber >= 0)
            childCanChangeMem = true;
    }

    const auto& detectInPlaceChildsNum = [](const std::vector<MKLDNNEdgePtr>& edges) -> size_t {
        size_t count = 0;
        for (const auto& edge : edges) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          for (const auto& edge : edges) {" << std::endl;
            auto childSPD = edge->getChild()->getSelectedPrimitiveDescriptor();
            int outNumber = edge->getOutputNum();
            if (childSPD->getConfig().outConfs.empty())
                count++;
            for (const auto conf : childSPD->getConfig().outConfs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              for (const auto conf : childSPD->getConfig().outConfs) {" << std::endl;
                if (conf.inPlace == outNumber)
                    count++;
            }
        }
        return count;
    };

    const auto portChildEdges = getParent()->getChildEdgesAtPort(inNumber);
    if (in_place && detectInPlaceChildsNum(portChildEdges) > 1 && childCanChangeMem)
        canBeInPlaceConflicts = true;
    if (!canBeInPlaceConflicts && in_place && !getParent()->getChildEdges().empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (!canBeInPlaceConflicts && in_place && !getParent()->getChildEdges().empty()) {" << std::endl;
        for (auto &p_edge_peer : portChildEdges) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          for (auto &p_edge_peer : portChildEdges) {" << std::endl;
            if (p_edge_peer.get() == this)
                continue;
            if (p_edge_peer->getChild()->getType() != Reorder && p_edge_peer->inPlace(LOOK_DOWN))
                canBeInPlaceConflicts = true;
        }
    }

    if (in_place) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (in_place) {" << std::endl;
        if (inNumber >= 0 && inNumber < parentSPD->getConfig().outConfs.size() && parentSPD->getConfig().outConfs[inNumber].inPlace >= 0 &&
            outNumber >= 0 && outNumber < childSPD->getConfig().inConfs.size() && childSPD->getConfig().inConfs[outNumber].inPlace >= 0)
            canBeInPlaceConflicts = true;
    }
    return canBeInPlaceConflicts || !MKLDNNExtensionUtils::initTensorsAreEqual(getInputDesc(), getOutputDesc());
}

InferenceEngine::TensorDesc MKLDNNEdge::getInputDesc() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  InferenceEngine::TensorDesc MKLDNNEdge::getInputDesc() {" << std::endl;
    if (inputDesc.getLayout() == InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (inputDesc.getLayout() == InferenceEngine::Layout::ANY) {" << std::endl;
        inputDesc = getSpecifiedInputDesc({});
    }
    return inputDesc;
}

InferenceEngine::TensorDesc MKLDNNEdge::getOutputDesc() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  InferenceEngine::TensorDesc MKLDNNEdge::getOutputDesc() {" << std::endl;
    if (outputDesc.getLayout() == InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (outputDesc.getLayout() == InferenceEngine::Layout::ANY) {" << std::endl;
        outputDesc = getSpecifiedOutputDesc({});
    }
    return outputDesc;
}

InferenceEngine::TensorDesc MKLDNNEdge::getDesc() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  InferenceEngine::TensorDesc MKLDNNEdge::getDesc() {" << std::endl;
    if (!MKLDNNExtensionUtils::initTensorsAreEqual(getInputDesc(), getOutputDesc()))
        THROW_IE_EXCEPTION << "Cannot get descriptor for edge: " << getParent()->getName() << "->"
                           << getChild()->getName();
    return getInputDesc();
}

int MKLDNNEdge::getInputNum() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  int MKLDNNEdge::getInputNum() {" << std::endl;
    return parent_port;
}

int MKLDNNEdge::getOutputNum() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  int MKLDNNEdge::getOutputNum() {" << std::endl;
    return child_port;
}

void MKLDNNEdge::allocate(const void* mem_ptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  void MKLDNNEdge::allocate(const void* mem_ptr) {" << std::endl;
    if (status != Status::NeedAllocation)
        return;

    if (memoryPtr)
        THROW_IE_EXCEPTION << "Unexpected behaviour: status == NeedAllocation but memory is already allocated.";

    auto inputDesc = getInputDesc();
    auto outputDesc = getOutputDesc();
    if (!MKLDNNExtensionUtils::initTensorsAreEqual(outputDesc, inputDesc) ||
            (inputDesc.getDims().size() > 0 && inputDesc.getDims()[0] != 1 && inputDesc != outputDesc))
        THROW_IE_EXCEPTION << "Cannot allocate memory. Nodes have primitive descriptors with different formats.";
    if (inputDesc.getLayout() == InferenceEngine::Layout::ANY)
        THROW_IE_EXCEPTION << "Cannot get input descriptor!";

    auto parentPtr = getParent();
    memoryPtr.reset(new MKLDNNMemory(parentPtr->getEngine()));
    memoryPtr->Create(MKLDNNMemoryDesc(inputDesc), mem_ptr, false);  // no pads zeroing
    status = Status::Allocated;
}

void MKLDNNEdge::changeStatus(MKLDNNEdge::Status state) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  void MKLDNNEdge::changeStatus(MKLDNNEdge::Status state) {" << std::endl;
    if (state == Status::NotAllocated) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (state == Status::NotAllocated) {" << std::endl;
        THROW_IE_EXCEPTION << "Incorrect behaviour! Use method sharedMemFrom()";
    }
    if (state == Status::Validated) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (state == Status::Validated) {" << std::endl;
        THROW_IE_EXCEPTION << "Incorrect behaviour! Use method validate()";
    }
    if (status != Status::Uninitialized && state == Status::NeedAllocation)
        return;
    if (status == Status::NotAllocated)
        memoryFromEdge.reset();
    status = state;
}

const MKLDNNDims& MKLDNNEdge::getDims() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  const MKLDNNDims& MKLDNNEdge::getDims() {" << std::endl;
    if (!dims.ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (!dims.ndims()) {" << std::endl;
        MKLDNNDims outDims;
        MKLDNNDims inDims;
        auto childPtr = getChild();
        auto parentPtr = getParent();

        int inNum = getOutputNum();
        if (inNum < 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (inNum < 0) {" << std::endl;
            THROW_IE_EXCEPTION << "Error cannot find input data for " << child.lock()->getName()
                               << " from " << parent.lock()->getName();
        }
        if (inNum < childPtr->inDims.size()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (inNum < childPtr->inDims.size()) {" << std::endl;
            outDims = childPtr->inDims[inNum];
        }

        int outNum = getInputNum();
        if (outNum < 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (outNum < 0) {" << std::endl;
            THROW_IE_EXCEPTION << "Error cannot find output data for " << parent.lock()->getName()
                               << " to " << child.lock()->getName();
        }
        if (outNum >= parentPtr->outDims.size())
            outNum = 0;
        if (outNum < parentPtr->outDims.size()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (outNum < parentPtr->outDims.size()) {" << std::endl;
            inDims = parentPtr->outDims[outNum];
        }

        if (inDims.ndims() && outDims.ndims() && inDims.ndims() != outDims.ndims() && inDims.size() != outDims.size())
            THROW_IE_EXCEPTION << "Nodes " << getParent()->getName() << " and " << getChild()->getName()
                               << " have incompatible dimensions!";

        if (outDims.ndims() != 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (outDims.ndims() != 0) {" << std::endl;
            dims = outDims;
        } else if (inDims.ndims() != 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          } else if (inDims.ndims() != 0) {" << std::endl;
            dims = inDims;
        } else {
            dims = MKLDNNDims({(size_t)1});
        }


        if (!(outDims.ndims() == 0 && inDims.ndims() == 0) && !dims.ndims())
            THROW_IE_EXCEPTION << "Cannot detect right dims for nodes " << getParent()->getName()
                               << " and " << getChild()->getName();
    }
    return dims;
}

bool MKLDNNEdge::nodeCanChangeDesc(const MKLDNNNodePtr &node) const {
    PrimitiveDescInfo * selectedPd = node->getSelectedPrimitiveDescriptor();
    if (selectedPd == nullptr)
        THROW_IE_EXCEPTION << "Primitive descriptor for node " << node->getName() << " is not selected.";

    for (auto &inputDesc : selectedPd->getConfig().inConfs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (auto &inputDesc : selectedPd->getConfig().inConfs) {" << std::endl;
        if (inputDesc.desc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (inputDesc.desc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
            return true;
        }
    }

    for (auto &outDesc : selectedPd->getConfig().outConfs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (auto &outDesc : selectedPd->getConfig().outConfs) {" << std::endl;
        if (outDesc.desc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (outDesc.desc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
            return true;
        }
    }

    MKLDNNDims inputDims;
    for (size_t i = 0; i < node->getParentEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (size_t i = 0; i < node->getParentEdges().size(); i++) {" << std::endl;
        if (inputDims.size() == 1 && inputDims.ndims() == 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (inputDims.size() == 1 && inputDims.ndims() == 0) {" << std::endl;
            inputDims = node->getParentEdgeAt(i)->getDims();
            continue;
        }

        if (inputDims.ndims() != node->getParentEdgeAt(i)->getDims().ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (inputDims.ndims() != node->getParentEdgeAt(i)->getDims().ndims()) {" << std::endl;
            return true;
        }
    }
    for (size_t i = 0; i < node->getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (size_t i = 0; i < node->getChildEdges().size(); i++) {" << std::endl;
        if (inputDims.size() == 1 && inputDims.ndims() == 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (inputDims.size() == 1 && inputDims.ndims() == 0) {" << std::endl;
            inputDims = node->getChildEdgeAt(i)->getDims();
            continue;
        }

        if (inputDims.ndims() != node->getChildEdgeAt(i)->getDims().ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (inputDims.ndims() != node->getChildEdgeAt(i)->getDims().ndims()) {" << std::endl;
            return true;
        }
    }

    return false;
}

/// In we have {any, any, any} -> {any} or {any} -> {any, any, any} or {any} -> {any} it means that
/// layer doesn't change memory format
/// We don't support {any, any, nchw} -> {any}
InferenceEngine::TensorDesc MKLDNNEdge::getSpecifiedInputDesc(std::map<mkldnn::memory::format, size_t> formats, size_t enterCountUp, size_t enterCountDown) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  InferenceEngine::TensorDesc MKLDNNEdge::getSpecifiedInputDesc(std::map<mkldnn::memory::format, size_t> formats, size_t enterCountUp, size_t enterCountDown) {" << std::endl;
    InferenceEngine::TensorDesc inDesc;

    if (inputDesc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (inputDesc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
        return inputDesc;
    }

    auto parentPtr = getParent();
    if (parentPtr->getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Primitive descriptor for node " << parentPtr->getName() << " is not selected.";

    int inputIdx = getInputNum();
    if (inputIdx < 0)
        THROW_IE_EXCEPTION << "Edge cannot be found for node" << parentPtr->getName() << ".";

    if (inputIdx >= parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size())
        inputIdx = 0;
    inDesc = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc;

    if (inDesc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (inDesc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
        return inDesc;
    }

    bool isFormatChanging = nodeCanChangeDesc(parentPtr);

    if (!isFormatChanging && inputIdx < parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() &&
            parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
        inDesc = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc;
        parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc = inDesc;
        return inDesc;
    }

    for (size_t i = 0; i < parentPtr->getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (size_t i = 0; i < parentPtr->getChildEdges().size(); i++) {" << std::endl;
        auto childEdge = parentPtr->getChildEdgeAt(i);
        auto child = childEdge->getChild();
        int childIdx = childEdge->getOutputNum();
        if (!child->getSelectedPrimitiveDescriptor() || childIdx < 0 ||
                childEdge->getDims().ndims() != getDims().ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:                  childEdge->getDims().ndims() != getDims().ndims()) {" << std::endl;
            continue;
        }
        if (child->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() <= childIdx)
            childIdx = 0;
        memory::format childInDesc = MKLDNNMemoryDesc(child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[childIdx].desc).getFormat();
        if (childInDesc != memory::any && childInDesc != memory::format_undef) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (childInDesc != memory::any && childInDesc != memory::format_undef) {" << std::endl;
            if (formats.find(childInDesc) == formats.end())
                formats[childInDesc] = 1;
            else
                formats[childInDesc] += 1;
            continue;
        }
        if (nodeCanChangeDesc(child))
            continue;

        if (enterCountUp < 2) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (enterCountUp < 2) {" << std::endl;
            childInDesc = MKLDNNMemoryDesc(childEdge->getSpecifiedOutputDesc(formats, enterCountUp, ++enterCountDown)).getFormat();
            if (childInDesc != memory::any && childInDesc != memory::format_undef) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              if (childInDesc != memory::any && childInDesc != memory::format_undef) {" << std::endl;
                if (formats.find(childInDesc) == formats.end())
                    formats[childInDesc] = 1;
                else
                    formats[childInDesc] += 1;
            }
        }
    }

    if (!isFormatChanging) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (!isFormatChanging) {" << std::endl;
        for (size_t i = 0; i < parentPtr->getParentEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          for (size_t i = 0; i < parentPtr->getParentEdges().size(); i++) {" << std::endl;
            auto parentEdge = parentPtr->getParentEdgeAt(i);
            auto parent = parentEdge->getParent();
            int parentIdx = parentEdge->getInputNum();
            if (!parent->getSelectedPrimitiveDescriptor() || parentIdx < 0 ||
                    parentEdge->getDims().ndims() != getDims().ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:                      parentEdge->getDims().ndims() != getDims().ndims()) {" << std::endl;
                continue;
            }
            if (parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= parentIdx) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              if (parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= parentIdx) {" << std::endl;
                parentIdx = 0;
            }
            memory::format parentOutDesc = MKLDNNMemoryDesc(parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[parentIdx].desc).getFormat();
            if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {" << std::endl;
                if (formats.find(parentOutDesc) == formats.end())
                    formats[parentOutDesc] = 1;
                else
                    formats[parentOutDesc] += 1;
                continue;
            }
            if (nodeCanChangeDesc(parent))
                continue;

            if (enterCountUp < 2) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              if (enterCountUp < 2) {" << std::endl;
                parentOutDesc = MKLDNNMemoryDesc(parentEdge->getSpecifiedInputDesc(formats, ++enterCountUp, enterCountDown)).getFormat();
                if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:                  if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {" << std::endl;
                    if (formats.find(parentOutDesc) == formats.end())
                        formats[parentOutDesc] = 1;
                    else
                        formats[parentOutDesc] += 1;
                }
            }
        }
    }

    size_t maxFormatCount = 0;
    memory::format desc =  MKLDNNMemory::GetPlainFormat(getDims());
    for (auto &it : formats) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (auto &it : formats) {" << std::endl;
        if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getDims(), it.first)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getDims(), it.first)) {" << std::endl;
            maxFormatCount = it.second;
            desc = it.first;
        }
    }

    auto inDataType = MKLDNNMemoryDesc(parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc).getDataType();
    parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc = MKLDNNMemoryDesc(getDims(), inDataType, desc);
    if (!isFormatChanging && inputIdx < parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() &&
            parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc.getLayout() == InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc.getLayout() == InferenceEngine::Layout::ANY) {" << std::endl;
        parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc =
                MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(getDims(), inDataType, desc));
    }

    return MKLDNNMemoryDesc(getDims(), inDataType, desc);
}

InferenceEngine::TensorDesc MKLDNNEdge::getSpecifiedOutputDesc(std::map<mkldnn::memory::format, size_t> formats, size_t enterCountUp, size_t enterCountDown) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  InferenceEngine::TensorDesc MKLDNNEdge::getSpecifiedOutputDesc(std::map<mkldnn::memory::format, size_t> formats, size_t enterCountUp, size_t enterCountDown) {" << std::endl;
    InferenceEngine::TensorDesc outDesc;

    if (outputDesc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (outputDesc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
        return outputDesc;
    }

    auto childPtr = getChild();
    auto parentPtr = getParent();

    if (childPtr->getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Primitive descriptor for node " << childPtr->getName() << " is not selected.";

    int outputIdx = getOutputNum();
    int inputIdx = getInputNum();
    if (outputIdx < 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (outputIdx < 0) {" << std::endl;
        THROW_IE_EXCEPTION << "Edge cannot be found for node" << childPtr->getName() << ".";
    }
    if (outputIdx >= childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size())
        outputIdx = 0;
    outDesc = childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc;

    if (outDesc.getLayout() != InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (outDesc.getLayout() != InferenceEngine::Layout::ANY) {" << std::endl;
        return outDesc;
    }

    if (inputIdx >= parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size())
        inputIdx = 0;

    bool isFormatChanging = nodeCanChangeDesc(childPtr);

    if ((!isFormatChanging && outputIdx < childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() &&
            childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc.getLayout() != InferenceEngine::Layout::ANY) ||
            (isFormatChanging && inputIdx >= 0 &&
                    parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc.getLayout() != InferenceEngine::Layout::ANY)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:                      parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc.getLayout() != InferenceEngine::Layout::ANY)) {" << std::endl;
        auto inputDataType = childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc.getPrecision();
        if (!isFormatChanging)
            outDesc = childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc;
        else
            outDesc = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc;
        childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc = InferenceEngine::TensorDesc(inputDataType, getDims().ToSizeVector(),
                                                    {outDesc.getBlockingDesc().getBlockDims(),
                                                     outDesc.getBlockingDesc().getOrder()});
        return childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc;
    }

    for (size_t i = 0; i < childPtr->getParentEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (size_t i = 0; i < childPtr->getParentEdges().size(); i++) {" << std::endl;
        auto parentEdge = childPtr->getParentEdgeAt(i);
        auto parent = parentEdge->getParent();
        int parentIdx = parentEdge->getInputNum();
        if (!parent->getSelectedPrimitiveDescriptor() || parentIdx < 0 ||
                parentEdge->getDims().ndims() != getDims().ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:                  parentEdge->getDims().ndims() != getDims().ndims()) {" << std::endl;
            continue;
        }
        if (parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= parentIdx) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= parentIdx) {" << std::endl;
            parentIdx = 0;
        }
        memory::format parentOutDesc = MKLDNNMemoryDesc(parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[parentIdx].desc).getFormat();
        if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {" << std::endl;
            if (formats.find(parentOutDesc) == formats.end())
                formats[parentOutDesc] = 1;
            else
                formats[parentOutDesc] += 1;
            continue;
        }
        if (nodeCanChangeDesc(parent))
            continue;

        if (enterCountDown < 2) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (enterCountDown < 2) {" << std::endl;
            parentOutDesc = MKLDNNMemoryDesc(parentEdge->getSpecifiedInputDesc(formats, ++enterCountUp, enterCountDown)).getFormat();
            if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {" << std::endl;
                if (formats.find(parentOutDesc) == formats.end())
                    formats[parentOutDesc] = 1;
                else
                    formats[parentOutDesc] += 1;
            }
        }
    }

    if (!isFormatChanging) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (!isFormatChanging) {" << std::endl;
        for (size_t i = 0; i < childPtr->getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          for (size_t i = 0; i < childPtr->getChildEdges().size(); i++) {" << std::endl;
            auto childEdge = childPtr->getChildEdgeAt(i);
            auto child = childEdge->getChild();
            int childIdx = childEdge->getOutputNum();
            if (!child->getSelectedPrimitiveDescriptor() || childIdx < 0 ||
                    childEdge->getDims().ndims() != getDims().ndims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:                      childEdge->getDims().ndims() != getDims().ndims()) {" << std::endl;
                continue;
            }
            if (child->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() <= childIdx) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              if (child->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() <= childIdx) {" << std::endl;
                childIdx = 0;
            }
            memory::format childInDesc = MKLDNNMemoryDesc(child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[childIdx].desc).getFormat();
            if (childInDesc != memory::any && childInDesc != memory::format_undef) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              if (childInDesc != memory::any && childInDesc != memory::format_undef) {" << std::endl;
                if (formats.find(childInDesc) == formats.end())
                    formats[childInDesc] = 1;
                else
                    formats[childInDesc] += 1;
                continue;
            }
            if (nodeCanChangeDesc(child))
                continue;

            if (enterCountDown < 2) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              if (enterCountDown < 2) {" << std::endl;
                childInDesc = MKLDNNMemoryDesc(childEdge->getSpecifiedOutputDesc(formats, enterCountUp, ++enterCountDown)).getFormat();
                if (childInDesc != memory::any && childInDesc != memory::format_undef) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:                  if (childInDesc != memory::any && childInDesc != memory::format_undef) {" << std::endl;
                    if (formats.find(childInDesc) == formats.end())
                        formats[childInDesc] = 1;
                    else
                        formats[childInDesc] += 1;
                }
            }
        }
    }

    size_t maxFormatCount = 0;
    memory::format format =  MKLDNNMemory::GetPlainFormat(getDims());
    for (auto &it : formats) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (auto &it : formats) {" << std::endl;
        if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getDims(), it.first)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getDims(), it.first)) {" << std::endl;
            maxFormatCount = it.second;
            format = it.first;
        }
    }

    auto inDataType = MKLDNNMemoryDesc(childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[getOutputNum()].desc).getDataType();
    childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc = MKLDNNMemoryDesc(getDims(), inDataType, format);
    if (!isFormatChanging && outputIdx < childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() &&
            childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc.getLayout() == InferenceEngine::Layout::ANY) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc.getLayout() == InferenceEngine::Layout::ANY) {" << std::endl;
        childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc =
                MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(getDims(), inDataType, format));
    }

    return childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc;
}

const MKLDNNMemory &MKLDNNEdge::getMemory() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  const MKLDNNMemory &MKLDNNEdge::getMemory() {" << std::endl;
    if (status == Status::NotAllocated) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (status == Status::NotAllocated) {" << std::endl;
        memoryPtr.reset(new MKLDNNMemory(getParent()->getEngine()));
        memoryPtr->Create(MKLDNNMemoryDesc(getDesc()), getSharedEdge()->getMemoryPtr()->GetData());
        memoryFromEdge.reset();
        changeStatus(Status::Allocated);
    }

    return *memoryPtr;
}

MKLDNNMemoryPtr &MKLDNNEdge::getMemoryPtr() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  MKLDNNMemoryPtr &MKLDNNEdge::getMemoryPtr() {" << std::endl;
    if (status == Status::NotAllocated) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (status == Status::NotAllocated) {" << std::endl;
        memoryPtr.reset(new MKLDNNMemory(getParent()->getEngine()));
        memoryPtr->Create(MKLDNNMemoryDesc(getDesc()), getSharedEdge()->getMemoryPtr()->GetData());
        memoryFromEdge.reset();
        changeStatus(Status::Allocated);
    }

    return memoryPtr;
}

InferenceEngine::Blob::Ptr MKLDNNEdge::getBlob() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  InferenceEngine::Blob::Ptr MKLDNNEdge::getBlob() {" << std::endl;
    if (!memoryPtr)
        THROW_IE_EXCEPTION << "Cannot get blob! Edge isn't initialized.";
    InferenceEngine::TensorDesc desc = getDesc();

    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        desc = InferenceEngine::TensorDesc(desc.getPrecision(), dims.ToSizeVector(), desc.getLayout());
    else
        desc = InferenceEngine::TensorDesc(desc.getPrecision(), dims.ToSizeVector(), desc.getBlockingDesc());

    return make_blob_with_precision(desc, memoryPtr->GetData());
}

void MKLDNNEdge::sharedMemFrom(const MKLDNNEdgePtr &edge) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  void MKLDNNEdge::sharedMemFrom(const MKLDNNEdgePtr &edge) {" << std::endl;
    memoryFromEdge = edge;
    status = Status::NotAllocated;
}

void MKLDNNEdge::validate() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  void MKLDNNEdge::validate() {" << std::endl;
    if (status == Status::Validated)
        return;
    getMemory();
    getParent();
    getChild();
    getDims();
    if (status != Status::Allocated) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (status != Status::Allocated) {" << std::endl;
        THROW_IE_EXCEPTION << "Error memory is not allocated!";
    }
    status = Status::Validated;
}

MKLDNNEdgePtr MKLDNNEdge::getSharedEdge() const {
    auto memoryFromEdgePtr = memoryFromEdge.lock();
    if (!memoryFromEdgePtr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (!memoryFromEdgePtr) {" << std::endl;
        THROW_IE_EXCEPTION << "Cannot get memory ptr for edge(" << getParent()->getName() << "->"
                           << getChild()->getName() << "). The pointer on the edge with memory is empty!";
    }
    return memoryFromEdgePtr;
}

void MKLDNNEdge::init() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  void MKLDNNEdge::init() {" << std::endl;
    if (status != Status::NeedAllocation && status != Status::Uninitialized)
        return;
    MKLDNNEdgePtr edgePtr = getBaseEdge();
    if (edgePtr.get() == this) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (edgePtr.get() == this) {" << std::endl;
        changeStatus(Status::NeedAllocation);
    } else {
        sharedMemFrom(edgePtr);
    }

    auto port = getInputNum();
    if (port < 0)
        return;
    auto edges_at_same_port = getParent()->getChildEdgesAtPort(static_cast<size_t>(port));
    for (auto edge : edges_at_same_port) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      for (auto edge : edges_at_same_port) {" << std::endl;
        if (edge->getStatus() != Status::NeedAllocation && edge->getStatus() != Status::Uninitialized) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (edge->getStatus() != Status::NeedAllocation && edge->getStatus() != Status::Uninitialized) {" << std::endl;
            if (edge->getSharedEdge() != edgePtr)
                THROW_IE_EXCEPTION << "Unsupported behavior. Cannot mark edge "
                                   << getParent()->getChildEdgeAt(0)->getParent()->getName() << "->"
                                   << getParent()->getChildEdgeAt(0)->getChild()->getName() << " as not allocated!";
        } else {
            if (edge != edgePtr)
                edge->sharedMemFrom(edgePtr);
        }
    }
}

/**
 * Should analyze graph node dependencies, inplace node information and return root memory(edge) it view on
 *
 * @param type some magic enum values... description needed
 * @return root of view-on-memory subgraph
 */
MKLDNNEdgePtr MKLDNNEdge::getBaseEdge(int look) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  MKLDNNEdgePtr MKLDNNEdge::getBaseEdge(int look) {" << std::endl;
    auto parentConfig = getParent()->getSelectedPrimitiveDescriptor()->getConfig();
    auto childConfig = getChild()->getSelectedPrimitiveDescriptor()->getConfig();
    int inputNum = getInputNum();
    int outputNum = getOutputNum();

    if (childConfig.inConfs[outputNum].inPlace >= 0 && parentConfig.outConfs[inputNum].inPlace >= 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (childConfig.inConfs[outputNum].inPlace >= 0 && parentConfig.outConfs[inputNum].inPlace >= 0) {" << std::endl;
        inputNum = getInputNum();
        return getParent()->getChildEdgeAt(inputNum);
    }

    if (childConfig.inConfs[outputNum].inPlace >= 0 && (look & LOOK_DOWN)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (childConfig.inConfs[outputNum].inPlace >= 0 && (look & LOOK_DOWN)) {" << std::endl;
        int next_port_idx = childConfig.inConfs[outputNum].inPlace;
        if (childConfig.outConfs[next_port_idx].inPlace >= 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (childConfig.outConfs[next_port_idx].inPlace >= 0) {" << std::endl;
            childConfig.outConfs[next_port_idx].inPlace = -1;
            getChild()->initDescriptor(childConfig);
        }

        auto ch_edges = getChild()->getChildEdgesAtPort(next_port_idx);
        auto &next_ch_edge = ch_edges[0];

        // Multiple connection to some out port
        // Will try to find inplace consumer
        for (auto &ch_edge : ch_edges) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          for (auto &ch_edge : ch_edges) {" << std::endl;
            auto &chch_conf = ch_edge->getChild()->getSelectedPrimitiveDescriptor()->getConfig();

            if (chch_conf.inConfs[ch_edge->getOutputNum()].inPlace >= 0)
                next_ch_edge = ch_edge;
        }
        return next_ch_edge->getBaseEdge(LOOK_DOWN);
    } else if (parentConfig.outConfs[inputNum].inPlace >= 0 && (look & LOOK_UP)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      } else if (parentConfig.outConfs[inputNum].inPlace >= 0 && (look & LOOK_UP)) {" << std::endl;
        int next_port_idx = parentConfig.outConfs[inputNum].inPlace;
        if (parentConfig.inConfs[next_port_idx].inPlace >= 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          if (parentConfig.inConfs[next_port_idx].inPlace >= 0) {" << std::endl;
            parentConfig.inConfs[next_port_idx].inPlace = -1;
            getParent()->initDescriptor(parentConfig);
        }
        return getParent()->getParentEdgesAtPort(next_port_idx)[0]->getBaseEdge(LOOK_UP);
    }

    auto edges_for_same_port = getParent()->getChildEdgesAtPort(inputNum);
    if (!(look & LOOK_NO_RECURRENT)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (!(look & LOOK_NO_RECURRENT)) {" << std::endl;
        for (auto edge : edges_for_same_port) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:          for (auto edge : edges_for_same_port) {" << std::endl;
            if (edge.get() != this) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:              if (edge.get() != this) {" << std::endl;
                auto base = edge->getBaseEdge(LOOK_BOTH | LOOK_NO_RECURRENT);
                if (base != edge && base != edges_for_same_port[0]) return base;
            }
        }
    }
    return edges_for_same_port[0];
}

bool MKLDNNEdge::inPlace(LOOK look) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:  bool MKLDNNEdge::inPlace(LOOK look) {" << std::endl;
    auto parentSPD = getParent()->getSelectedPrimitiveDescriptor();
    auto childSPD = getChild()->getSelectedPrimitiveDescriptor();
    if (!parentSPD || !childSPD)
        THROW_IE_EXCEPTION << "Cannot make a decision about reorder. Primitive descriptors weren't selected.";
    int inputNum = getInputNum();
    int outputNum = getOutputNum();
    if (inputNum >= parentSPD->getConfig().outConfs.size())
        inputNum = 0;
    if (outputNum >= childSPD->getConfig().inConfs.size())
        outputNum = 0;

    if (look & LOOK_UP) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (look & LOOK_UP) {" << std::endl;
        if (parentSPD->getConfig().outConfs[inputNum].inPlace >= 0)
            return true;
    }
    if (look & LOOK_DOWN) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_edge.cpp:      if (look & LOOK_DOWN) {" << std::endl;
        if (childSPD->getConfig().inConfs[outputNum].inPlace >= 0)
            return true;
    }
    return false;
}

}  // namespace MKLDNNPlugin
