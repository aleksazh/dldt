#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_infer_request.h"
#include "mkldnn_extension_utils.h"
#include "mkldnn_streams.h"
#include <vector>
#include <string>
#include <map>
#include <blob_factory.hpp>
#include <nodes/mkldnn_concat_node.h>
#include <nodes/mkldnn_split_node.h>
#include <ie_compound_blob.h>

MKLDNNPlugin::MKLDNNInferRequest::MKLDNNInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                                     InferenceEngine::OutputsDataMap networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          : InferRequestInternal(networkInputs, networkOutputs) {" << std::endl;}


template <typename T> void MKLDNNPlugin::MKLDNNInferRequest::pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:  template <typename T> void MKLDNNPlugin::MKLDNNInferRequest::pushInput(const std::string& inputName, InferenceEngine::Blob::Ptr& inputBlob) {" << std::endl;
    InferenceEngine::TBlob<T> *in_f = dynamic_cast<InferenceEngine::TBlob<T> *>(inputBlob.get());

    if (in_f == nullptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      if (in_f == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "Input data precision not supported. Expected float.";
    }

    if (in_f->readOnly() == nullptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      if (in_f->readOnly() == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "Input data was not allocated.";
    }

    graph->PushInputData(inputName, inputBlob);
}

void MKLDNNPlugin::MKLDNNInferRequest::InferImpl() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:  void MKLDNNPlugin::MKLDNNInferRequest::InferImpl() {" << std::endl;
    IE_PROFILING_AUTO_SCOPE(MKLDNN_INFER)
    if (!graph || !graph->IsReady()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      if (!graph || !graph->IsReady()) {" << std::endl;
        THROW_IE_EXCEPTION << "Network not loaded.";
    }
    auto infer = [this] {
        // execute input pre-processing.
        execDataPreprocessing(_inputs);

        changeDefaultPtr();
        // need to retain converted blobs until infer finish
        std::vector<InferenceEngine::Blob::Ptr> convertedInputs;
        for (auto input : _inputs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          for (auto input : _inputs) {" << std::endl;
            if (!_networkInputs[input.first]) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:              if (!_networkInputs[input.first]) {" << std::endl;
                THROW_IE_EXCEPTION <<
                                   "input blobs map contains not registered during IInferencePlugin::LoadNetwork blob with name "
                                   << input.first;
            }

            InferenceEngine::Blob::Ptr iconv;
            InferenceEngine::TBlob<float> *in_f = nullptr;
            switch (input.second->getTensorDesc().getPrecision()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:              switch (input.second->getTensorDesc().getPrecision()) {" << std::endl;
                case InferenceEngine::Precision::FP32:
                    pushInput<float>(input.first, input.second);
                    break;
                case InferenceEngine::Precision::I32:
                    pushInput<int32_t>(input.first, input.second);
                    break;
                case InferenceEngine::Precision::I8:
                    pushInput<int8_t>(input.first, input.second);
                    break;
                case InferenceEngine::Precision::U16:
                    // U16 is unsupported by mkldnn, so here we convert the blob and send FP32
                    iconv = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                                      input.second->getTensorDesc().getDims(),
                                                                      input.second->getTensorDesc().getLayout()});
                    convertedInputs.push_back(iconv);
                    iconv->allocate();
                    in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                    if (in_f == nullptr)
                        THROW_IE_EXCEPTION << "Cannot get TBlob";
                    IE_SUPPRESS_DEPRECATED_START
                    InferenceEngine::copyToFloat<uint16_t>(in_f->data(), input.second.get());
                    IE_SUPPRESS_DEPRECATED_END
                    pushInput<float>(input.first, iconv);
                    break;
                case InferenceEngine::Precision::I16:
                    if (graph->hasMeanImageFor(input.first)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                      if (graph->hasMeanImageFor(input.first)) {" << std::endl;
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                                          input.second->getTensorDesc().getDims(),
                                                                          input.second->getTensorDesc().getLayout()});
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        if (in_f == nullptr)
                            THROW_IE_EXCEPTION << "Cannot get TBlob";
                        IE_SUPPRESS_DEPRECATED_START
                        InferenceEngine::copyToFloat<int16_t>(in_f->data(), input.second.get());
                        IE_SUPPRESS_DEPRECATED_END
                        pushInput<float>(input.first, iconv);
                    } else {
                        // Instead we can send I16 directly
                        pushInput<int16_t>(input.first, input.second);
                    }
                    break;
                case InferenceEngine::Precision::U8:
                    if (graph->hasMeanImageFor(input.first)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                      if (graph->hasMeanImageFor(input.first)) {" << std::endl;
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32,
                                                                          input.second->getTensorDesc().getDims(),
                                                                          input.second->getTensorDesc().getLayout()});
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        if (in_f == nullptr)
                            THROW_IE_EXCEPTION << "Cannot get TBlob";
                        IE_SUPPRESS_DEPRECATED_START
                        InferenceEngine::copyToFloat<uint8_t>(in_f->data(), input.second.get());
                        IE_SUPPRESS_DEPRECATED_END
                        pushInput<float>(input.first, iconv);
                    } else {
                        // Instead we can send I8 directly
                        pushInput<uint8_t>(input.first, input.second);
                    }
                    break;
                default:
                    THROW_IE_EXCEPTION << "Unsupported input precision " << input.second->getTensorDesc().getPrecision();
            }
        }
        graph->Infer(m_curBatch);
        graph->PullOutputData(_outputs);
    };
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    auto_scope_observing observer(graph->ptrObserver);
    // a TBB arena is made "this" for Infer call via executing lambda for the arena
    graph->ptrArena->execute([&] { infer(); });
#else
    infer();
#endif
}

void MKLDNNPlugin::MKLDNNInferRequest::GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";
    graph->GetPerfData(perfMap);
}

void MKLDNNPlugin::MKLDNNInferRequest::GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:  void MKLDNNPlugin::MKLDNNInferRequest::GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) {" << std::endl;
    IE_PROFILING_AUTO_SCOPE(GetBlob)
    if (!graph || !graph->IsReady())
        THROW_IE_EXCEPTION << "Graph is not ready!";

    InferenceEngine::BlobMap blobs;
    graph->getInputBlobs(blobs);

    if (blobs.find(name) != blobs.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      if (blobs.find(name) != blobs.end()) {" << std::endl;
        // ROI blob is returned only if it was set previously.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (it != _preProcData.end()) {" << std::endl;
            data = it->second->getRoiBlob();
            return;
        }

        if (_inputs.find(name) != _inputs.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (_inputs.find(name) != _inputs.end()) {" << std::endl;
            data = _inputs[name];
            checkBlob(data, name, true);
            return;
        }

        InferenceEngine::TensorDesc desc = blobs[name]->getTensorDesc();
        InferenceEngine::Precision originPrecision = blobs[name]->getTensorDesc().getPrecision();
        if (_networkInputs.find(name) != _networkInputs.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (_networkInputs.find(name) != _networkInputs.end()) {" << std::endl;
            InferenceEngine::Layout l = _networkInputs[name]->getLayout();
            InferenceEngine::Precision p = _networkInputs[name]->getPrecision();
            InferenceEngine::SizeVector dims = _networkInputs[name]->getTensorDesc().getDims();

            desc = InferenceEngine::TensorDesc(p, dims, l);
        }

        _inputs[name] = make_blob_with_precision(desc);
        _inputs[name]->allocate();
        if (desc.getPrecision() == originPrecision &&
                graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                  graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {" << std::endl;
            externalPtr[name] = _inputs[name]->buffer();
        }
        data = _inputs[name];
        checkBlob(data, name, true);
        return;
    }
    blobs.clear();
    graph->getOutputBlobs(blobs);

    if (blobs.find(name) != blobs.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      if (blobs.find(name) != blobs.end()) {" << std::endl;
        if (_outputs.find(name) != _outputs.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (_outputs.find(name) != _outputs.end()) {" << std::endl;
            data = _outputs[name];
            checkBlob(data, name, false);
            return;
        }

        _outputs[name] = make_blob_with_precision(blobs[name]->getTensorDesc());
        _outputs[name]->allocate();
        if (blobs[name]->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                !graph->getProperty().batchLimit) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                  !graph->getProperty().batchLimit) {" << std::endl;
            externalPtr[name] = _outputs[name]->buffer();
        }
        data = _outputs[name];
        checkBlob(data, name, false);
        return;
    }
    THROW_IE_EXCEPTION << "Cannot find blob with name: " << name;
}

void MKLDNNPlugin::MKLDNNInferRequest::SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:  void MKLDNNPlugin::MKLDNNInferRequest::SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) {" << std::endl;
    IE_PROFILING_AUTO_SCOPE(SetBlob)
    if (name == nullptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      if (name == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    if (!data)
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    const bool compoundBlobPassed = data->is<CompoundBlob>();
    if (!compoundBlobPassed && data->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
    if (data->size() == 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      if (data->size() == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "Input data is empty. Input name: \'" << name << "\'";
    }

    InferenceEngine::InputInfo::Ptr foundInput;
    InferenceEngine::DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {" << std::endl;
        if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (foundInput->getPrecision() != data->getTensorDesc().getPrecision()) {" << std::endl;
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set Blob with precision "
                               << data->getTensorDesc().getPrecision();
        }

        const bool preProcRequired = preProcessingRequired(foundInput, data);
        if (compoundBlobPassed && !preProcRequired) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (compoundBlobPassed && !preProcRequired) {" << std::endl;
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (preProcRequired) {" << std::endl;
            if (_preProcData.find(name) == _preProcData.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:              if (_preProcData.find(name) == _preProcData.end()) {" << std::endl;
                _preProcData.emplace(name, CreatePreprocDataHelper());
            }
            _preProcData[name]->isApplicable(data, _inputs[name]);
            // Stores the given blob as ROI blob. It will be used to fill in network input during
            // pre-processing
            _preProcData[name]->setRoiBlob(data);
        } else {
            size_t inputSize = foundInput->getTensorDesc().getLayout() != SCALAR
                ? InferenceEngine::details::product(foundInput->getTensorDesc().getDims())
                : 1;
            if (dataSize != inputSize) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:              if (dataSize != inputSize) {" << std::endl;
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                                   << dataSize << "!=" << inputSize << ").";
            }

            if (foundInput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:              if (foundInput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {" << std::endl;
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set input Blob. Dimensions mismatch.";
            }

            if (data->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                  graph->_meanImages.find(name) == graph->_meanImages.end() && !graph->getProperty().batchLimit) {" << std::endl;
                externalPtr[name] = data->buffer();
            } else if (externalPtr.find(name) != externalPtr.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:              } else if (externalPtr.find(name) != externalPtr.end()) {" << std::endl;
                externalPtr.erase(name);
            }
            _inputs[name] = data;
        }
    } else {
        if (compoundBlobPassed) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (compoundBlobPassed) {" << std::endl;
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "cannot set compound blob: supported only for input pre-processing";
        }
        size_t outputSize = foundOutput->getTensorDesc().getLayout() != SCALAR
            ? InferenceEngine::details::product(foundOutput->getDims())
            : 1;
        if (dataSize != outputSize) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (dataSize != outputSize) {" << std::endl;
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (foundOutput->getTensorDesc().getDims() != data->getTensorDesc().getDims()) {" << std::endl;
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set output Blob. Dimensions mismatch.";
        }
        if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (foundOutput->getPrecision() != data->getTensorDesc().getPrecision()) {" << std::endl;
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user output precision";
        }
        if (data->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32 &&
                !graph->getProperty().batchLimit) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                  !graph->getProperty().batchLimit) {" << std::endl;
            externalPtr[name] = data->buffer();
        } else if (externalPtr.find(name) != externalPtr.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          } else if (externalPtr.find(name) != externalPtr.end()) {" << std::endl;
            externalPtr.erase(name);
        }
        _outputs[name] = data;
    }
}

static inline void changeEdgePtr(const MKLDNNPlugin::MKLDNNEdgePtr &edge, void *newPtr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:  static inline void changeEdgePtr(const MKLDNNPlugin::MKLDNNEdgePtr &edge, void *newPtr) {" << std::endl;
    edge->getMemory().GetPrimitivePtr()->set_data_handle(newPtr);
}

void MKLDNNPlugin::MKLDNNInferRequest::changeDefaultPtr() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:  void MKLDNNPlugin::MKLDNNInferRequest::changeDefaultPtr() {" << std::endl;
    for (auto& it : externalPtr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      for (auto& it : externalPtr) {" << std::endl;
        auto input = graph->inputNodes.find(it.first);
        if (input != graph->inputNodes.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (input != graph->inputNodes.end()) {" << std::endl;
            if (input->second->getChildEdgeAt(0)->getMemory().GetPrimitive().get_data_handle() == it.second)
                continue;
            // Input cannot be in-place with other primitives
            bool canBeInPlace = true;
            for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:              for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {" << std::endl;
                auto& child = input->second->getChildEdgeAt(i)->getChild();
                if (child->isConstant())
                    canBeInPlace = false;
#if defined(COMPILED_CPU_MKLDNN_CONCAT_NODE)
                auto* concat = dynamic_cast<MKLDNNConcatNode *>(child.get());
                if (canBeInPlace && concat && concat->isOptimized())
                    canBeInPlace = false;
#endif
                // Cannot be in-place before split because split is using different ptrs without offsets
#if defined(COMPILED_CPU_MKLDNN_SPLIT_NODE)
                auto* split = dynamic_cast<MKLDNNSplitNode *>(child.get());
                if (canBeInPlace && split)
                    canBeInPlace = false;
#endif

                if (child->isInplace())
                    canBeInPlace = false;
                for (size_t j = 0; canBeInPlace && j < child->getChildEdges().size(); j++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                  for (size_t j = 0; canBeInPlace && j < child->getChildEdges().size(); j++) {" << std::endl;
                    if (child->getChildEdgeAt(j)->getMemory().GetPrimitive().get_data_handle() ==
                            input->second->getChildEdgeAt(i)->getMemory().GetPrimitive().get_data_handle())
                        canBeInPlace = false;
                }
            }
            for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:              for (size_t i = 0; canBeInPlace && i < input->second->getChildEdges().size(); i++) {" << std::endl;
                changeEdgePtr(input->second->getChildEdgeAt(i), it.second);
            }
            continue;
        }

        MKLDNNNodePtr output;
        for (auto& out : graph->outputNodes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          for (auto& out : graph->outputNodes) {" << std::endl;
            if (out->getName() == "out_" + it.first) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:              if (out->getName() == 'out_' + it.first) {" << std::endl;
                output = out;
                break;
            }
        }
        if (output) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:          if (output) {" << std::endl;
            if (output->getParentEdgeAt(0)->getMemory().GetPrimitive().get_data_handle() == it.second)
                continue;
            bool canBeInPlace = true;
            void * defaultPtr = output->getParentEdgeAt(0)->getMemory().GetPrimitivePtr()->get_data_handle();
            // Cannot be in-place after concat because concat is using different ptrs without offsets
            auto parent = output->getParentEdgeAt(0)->getParent();
            MKLDNNNodePtr previousParent;
            do {
                previousParent = parent;
                if (parent->getChildEdges().size() != 1 || parent->isConstant() || parent->isInplace()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                  if (parent->getChildEdges().size() != 1 || parent->isConstant() || parent->isInplace()) {" << std::endl;
                    canBeInPlace = false;
                    break;
                }

                for (size_t i = 0; i < parent->getParentEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                  for (size_t i = 0; i < parent->getParentEdges().size(); i++) {" << std::endl;
                    if (parent->getParentEdgeAt(i)->getMemory().GetPrimitivePtr()->get_data_handle() == defaultPtr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:                      if (parent->getParentEdgeAt(i)->getMemory().GetPrimitivePtr()->get_data_handle() == defaultPtr) {" << std::endl;
                        parent = parent->getParentEdgeAt(i)->getParent();
                        break;
                    }
                }
            } while (previousParent != parent);
            if (canBeInPlace)
                changeEdgePtr(output->getParentEdgeAt(0), it.second);
            continue;
        }
        THROW_IE_EXCEPTION << "Cannot find input/output blob: " << it.first;
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::SetGraph(const MKLDNNPlugin::MKLDNNGraph::Ptr &graph) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:  void MKLDNNPlugin::MKLDNNInferRequest::SetGraph(const MKLDNNPlugin::MKLDNNGraph::Ptr &graph) {" << std::endl;
    this->graph = graph;

    InferenceEngine::BlobMap blobs;
    this->graph->getInputBlobs(blobs);
    for (const auto& it : blobs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      for (const auto& it : blobs) {" << std::endl;
        InferenceEngine::Blob::Ptr blob;
        GetBlob(it.first.c_str(), blob);
    }
    blobs.clear();
    this->graph->getOutputBlobs(blobs);
    for (const auto& it : blobs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      for (const auto& it : blobs) {" << std::endl;
        InferenceEngine::Blob::Ptr blob;
        GetBlob(it.first.c_str(), blob);
    }
}

void MKLDNNPlugin::MKLDNNInferRequest::SetBatch(int new_batch) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:  void MKLDNNPlugin::MKLDNNInferRequest::SetBatch(int new_batch) {" << std::endl;
    if (!graph->getProperty().enableDynamicBatch)
        THROW_IE_EXCEPTION << "Dynamic batch is not enabled.";

    if (new_batch < 1 || new_batch > graph->getProperty().batchLimit) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_infer_request.cpp:      if (new_batch < 1 || new_batch > graph->getProperty().batchLimit) {" << std::endl;
        THROW_IE_EXCEPTION << "Invalid dynamic batch size " << new_batch <<
            " for this request.";
    }

    m_curBatch = new_batch;
}
