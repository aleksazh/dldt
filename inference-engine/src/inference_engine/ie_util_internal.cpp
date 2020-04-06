// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_util_internal.hpp"

#include <ie_layers.h>

#include <cassert>
#include <deque>
#include <iomanip>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>

#include "cpp/ie_plugin_cpp.hpp"
#include "details/caseless.hpp"
#include "details/ie_cnn_network_tools.h"
#include "details/os/os_filesystem.hpp"
#include "file_utils.h"
#include "graph_tools.hpp"
#include "ie_icnn_network_stats.hpp"
#include "net_pass.h"
#include "precision_utils.h"

using std::string;

namespace InferenceEngine {

std::exception_ptr& CurrentException() {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  std::exception_ptr& CurrentException() {" << std::endl;
    static thread_local std::exception_ptr currentException = nullptr;
    return currentException;
}

using namespace details;

DataPtr cloneData(const InferenceEngine::Data& source) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  DataPtr cloneData(const InferenceEngine::Data& source) {" << std::endl;
    auto cloned = std::make_shared<InferenceEngine::Data>(source);
    if (cloned != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      if (cloned != nullptr) {" << std::endl;
        cloned->getCreatorLayer().reset();
        cloned->getInputTo().clear();
    }
    return cloned;
}

namespace {
template <typename T>
CNNLayerPtr layerCloneImpl(const CNNLayer* source) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  CNNLayerPtr layerCloneImpl(const CNNLayer* source) {" << std::endl;
    auto layer = dynamic_cast<const T*>(source);
    if (nullptr != layer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      if (nullptr != layer) {" << std::endl;
        auto newLayer = std::make_shared<T>(*layer);
        newLayer->_fusedWith = nullptr;
        newLayer->outData.clear();
        newLayer->insData.clear();
        return std::static_pointer_cast<CNNLayer>(newLayer);
    }
    return nullptr;
}

/* Make this function explicit for TensorIterator layer
 * because of specific handling of the body field */
template <>
CNNLayerPtr layerCloneImpl<TensorIterator>(const CNNLayer* source) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  CNNLayerPtr layerCloneImpl<TensorIterator>(const CNNLayer* source) {" << std::endl;
    auto layer = dynamic_cast<const TensorIterator*>(source);
    if (nullptr != layer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      if (nullptr != layer) {" << std::endl;
        auto newLayer = std::make_shared<TensorIterator>(*layer);
        newLayer->_fusedWith = nullptr;
        newLayer->outData.clear();
        newLayer->insData.clear();

        newLayer->body = NetPass::CopyTIBody(newLayer->body);

        return std::static_pointer_cast<CNNLayer>(newLayer);
    }
    return nullptr;
}

}  // namespace

CNNLayerPtr clonelayer(const CNNLayer& source) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  CNNLayerPtr clonelayer(const CNNLayer& source) {" << std::endl;
    using fptr = CNNLayerPtr (*)(const CNNLayer*);
    // Most derived layers must go first in this list
    static const fptr cloners[] = {&layerCloneImpl<ScatterLayer>,
                                   &layerCloneImpl<NonMaxSuppressionLayer>,
                                   &layerCloneImpl<SelectLayer>,
                                   &layerCloneImpl<BatchNormalizationLayer>,
                                   &layerCloneImpl<TopKLayer>,
                                   &layerCloneImpl<PowerLayer>,
                                   &layerCloneImpl<ScaleShiftLayer>,
                                   &layerCloneImpl<PReLULayer>,
                                   &layerCloneImpl<TileLayer>,
                                   &layerCloneImpl<ReshapeLayer>,
                                   &layerCloneImpl<CropLayer>,
                                   &layerCloneImpl<EltwiseLayer>,
                                   &layerCloneImpl<GemmLayer>,
                                   &layerCloneImpl<PadLayer>,
                                   &layerCloneImpl<GatherLayer>,
                                   &layerCloneImpl<StridedSliceLayer>,
                                   &layerCloneImpl<ShuffleChannelsLayer>,
                                   &layerCloneImpl<DepthToSpaceLayer>,
                                   &layerCloneImpl<SpaceToDepthLayer>,
                                   &layerCloneImpl<SparseFillEmptyRowsLayer>,
                                   &layerCloneImpl<SparseSegmentReduceLayer>,
                                   &layerCloneImpl<ExperimentalSparseWeightedReduceLayer>,
                                   &layerCloneImpl<SparseToDenseLayer>,
                                   &layerCloneImpl<BucketizeLayer>,
                                   &layerCloneImpl<ReverseSequenceLayer>,
                                   &layerCloneImpl<RangeLayer>,
                                   &layerCloneImpl<FillLayer>,
                                   &layerCloneImpl<BroadcastLayer>,
                                   &layerCloneImpl<MathLayer>,
                                   &layerCloneImpl<ReduceLayer>,
                                   &layerCloneImpl<ClampLayer>,
                                   &layerCloneImpl<ReLULayer>,
                                   &layerCloneImpl<SoftMaxLayer>,
                                   &layerCloneImpl<GRNLayer>,
                                   &layerCloneImpl<MVNLayer>,
                                   &layerCloneImpl<NormLayer>,
                                   &layerCloneImpl<SplitLayer>,
                                   &layerCloneImpl<ConcatLayer>,
                                   &layerCloneImpl<FullyConnectedLayer>,
                                   &layerCloneImpl<PoolingLayer>,
                                   &layerCloneImpl<DeconvolutionLayer>,
                                   &layerCloneImpl<DeformableConvolutionLayer>,
                                   &layerCloneImpl<ConvolutionLayer>,
                                   &layerCloneImpl<TensorIterator>,
                                   &layerCloneImpl<RNNSequenceLayer>,
                                   &layerCloneImpl<LSTMCell>,
                                   &layerCloneImpl<GRUCell>,
                                   &layerCloneImpl<RNNCell>,
                                   &layerCloneImpl<QuantizeLayer>,
                                   &layerCloneImpl<BinaryConvolutionLayer>,
                                   &layerCloneImpl<WeightableLayer>,
                                   &layerCloneImpl<OneHotLayer>,
                                   &layerCloneImpl<CNNLayer>,
                                   &layerCloneImpl<UniqueLayer>};
    for (auto cloner : cloners) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      for (auto cloner : cloners) {" << std::endl;
        auto cloned = cloner(&source);
        if (nullptr != cloned) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          if (nullptr != cloned) {" << std::endl;
            return cloned;
        }
    }
    assert(!"All layers derived from CNNLayer so we must never get here");
    return nullptr;  // Silence "control may reach end of non-void function" warning
}

details::CNNNetworkImplPtr cloneNet(const ICNNNetwork& network) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  details::CNNNetworkImplPtr cloneNet(const ICNNNetwork& network) {" << std::endl;
    std::vector<CNNLayerPtr> layers;
    details::CNNNetworkIterator i(const_cast<ICNNNetwork*>(&network));
    while (i != details::CNNNetworkIterator()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      while (i != details::CNNNetworkIterator()) {" << std::endl;
        layers.push_back(*i);
        i++;
    }

    InferenceEngine::ICNNNetworkStats* pstatsSrc = nullptr;
    if (StatusCode::OK != network.getStats(&pstatsSrc, nullptr)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      if (StatusCode::OK != network.getStats(&pstatsSrc, nullptr)) {" << std::endl;
        pstatsSrc = nullptr;
    }
    // copy of the network
    details::CNNNetworkImplPtr net = cloneNet(layers, pstatsSrc);
    // going over output layers and aligning output ports and outputs
    OutputsDataMap outputs;
    network.getOutputsInfo(outputs);
    OutputsDataMap outputInfo;
    net->getOutputsInfo(outputInfo);
    for (auto o : outputs) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      for (auto o : outputs) {" << std::endl;
        auto it = outputInfo.find(o.first);
        if (it != outputInfo.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          if (it != outputInfo.end()) {" << std::endl;
            outputInfo.erase(it);
        } else {
            net->addOutput(o.first);
        }
    }
    // remove output ports which unconnected with outputs
    for (auto o : outputInfo) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      for (auto o : outputInfo) {" << std::endl;
        net->removeOutput(o.first);
    }
    net->setPrecision(network.getPrecision());
    net->setName(network.getName());

    InputsDataMap externalInputsData;
    network.getInputsInfo(externalInputsData);

    InputsDataMap clonedInputs;
    net->getInputsInfo(clonedInputs);
    for (auto&& it : externalInputsData) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      for (auto&& it : externalInputsData) {" << std::endl;
        auto inp = clonedInputs.find(it.first);
        if (inp != clonedInputs.end() && nullptr != inp->second) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          if (inp != clonedInputs.end() && nullptr != inp->second) {" << std::endl;
            inp->second->setPrecision(it.second->getPrecision());
            inp->second->getPreProcess() = it.second->getPreProcess();
        }
    }

    return net;
}

details::CNNNetworkImplPtr cloneNet(const std::vector<CNNLayerPtr>& layers, const ICNNNetworkStats* networkStats) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  details::CNNNetworkImplPtr cloneNet(const std::vector<CNNLayerPtr>& layers, const ICNNNetworkStats* networkStats) {" << std::endl;
    auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>();

    // Src to cloned data map
    std::unordered_map<InferenceEngine::DataPtr, InferenceEngine::DataPtr> dataMap;
    // Cloned to src data map
    std::unordered_map<InferenceEngine::DataPtr, InferenceEngine::DataPtr> clonedDataMap;
    std::vector<InferenceEngine::DataPtr> clonedDatas;

    auto createDataImpl = [&](const InferenceEngine::DataPtr& data) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      auto createDataImpl = [&](const InferenceEngine::DataPtr& data) {" << std::endl;
        assert(nullptr != data);
        if (!contains(dataMap, data)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          if (!contains(dataMap, data)) {" << std::endl;
            auto clonedData = cloneData(*data);
            dataMap[data] = clonedData;
            clonedDataMap[clonedData] = data;
            clonedDatas.push_back(clonedData);
            net->getData(clonedData->getName()) = clonedData;
            return clonedData;
        }
        return dataMap[data];
    };

    auto cloneLayerImpl = [&](const CNNLayer& srcLayer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      auto cloneLayerImpl = [&](const CNNLayer& srcLayer) {" << std::endl;
        CNNLayerPtr clonedLayer = clonelayer(srcLayer);
        clonedLayer->_fusedWith = nullptr;
        // We will need to reconstruct all connections in new graph
        clonedLayer->outData.clear();
        clonedLayer->insData.clear();
        net->addLayer(clonedLayer);
        return clonedLayer;
    };

    for (auto&& srcLayer : layers) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      for (auto&& srcLayer : layers) {" << std::endl;
        CNNLayerPtr clonedLayer = cloneLayerImpl(*srcLayer);
        for (auto&& src : srcLayer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          for (auto&& src : srcLayer->insData) {" << std::endl;
            auto data = src.lock();
            auto clonedData = createDataImpl(data);

            string inputName;
            // Find input name
            for (auto&& inp : data->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              for (auto&& inp : data->getInputTo()) {" << std::endl;
                if (srcLayer == inp.second) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:                  if (srcLayer == inp.second) {" << std::endl;
                    inputName = inp.first;
                    break;
                }
            }
            assert(!inputName.empty());
            clonedData->getInputTo().insert({inputName, clonedLayer});
            clonedLayer->insData.push_back(clonedData);
        }

        for (auto&& data : srcLayer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          for (auto&& data : srcLayer->outData) {" << std::endl;
            auto clonedData = createDataImpl(data);
            clonedData->getCreatorLayer() = clonedLayer;
            clonedLayer->outData.push_back(clonedData);
            for (auto&& inp : data->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              for (auto&& inp : data->getInputTo()) {" << std::endl;
                auto layer = inp.second;
                // TODO(amalyshe) is it the best place to check priorbox and remove
                // such edge from outputs?
                if (std::find(layers.begin(), layers.end(), layer) == layers.end() &&
                    !(CaselessEq<string>()(layer->type, "priorbox") ||
                      CaselessEq<string>()(layer->type, "PriorBoxClustered"))) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:                        CaselessEq<string>()(layer->type, 'PriorBoxClustered'))) {" << std::endl;
                    net->addOutput(data->getName());
                    break;
                }
            }
        }
    }

    for (auto&& data : clonedDatas) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      for (auto&& data : clonedDatas) {" << std::endl;
        auto layer = data->getCreatorLayer().lock();
        // create an artificial input layer because logic in some algorithms rely
        // on existence of these layers in the network
        if (nullptr == layer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          if (nullptr == layer) {" << std::endl;
            assert(contains(clonedDataMap, data));
            auto originalData = clonedDataMap[data];
            assert(nullptr != originalData);

            if (auto originalLayer = originalData->getCreatorLayer().lock()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              if (auto originalLayer = originalData->getCreatorLayer().lock()) {" << std::endl;
                if (CaselessEq<string>()(originalLayer->type, "input") ||
                    CaselessEq<string>()(originalLayer->type, "const") ||
                    CaselessEq<string>()(originalLayer->type, "memory")) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:                      CaselessEq<string>()(originalLayer->type, 'memory')) {" << std::endl;
                    layer = cloneLayerImpl(*originalLayer);
                    layer->outData.push_back(data);
                    data->getCreatorLayer() = layer;
                }
            }

            if (nullptr == layer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              if (nullptr == layer) {" << std::endl;
                LayerParams params;
                params.name = data->getName();
                params.precision = data->getPrecision();
                params.type = "Input";
                layer = std::make_shared<CNNLayer>(params);
                // this place should be transactional
                layer->outData.push_back(data);
                data->getCreatorLayer() = layer;
                net->addLayer(layer);
            }
        }
        if (CaselessEq<string>()(layer->type, "input")) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          if (CaselessEq<string>()(layer->type, 'input')) {" << std::endl;
            auto input = std::make_shared<InferenceEngine::InputInfo>();
            input->setInputData(data);
            net->setInputInfo(input);
        }
    }

    net->resolveOutput();

    // cloning of statistics
    InferenceEngine::ICNNNetworkStats* pstatsTarget = nullptr;
    if (networkStats != nullptr && !networkStats->isEmpty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      if (networkStats != nullptr && !networkStats->isEmpty()) {" << std::endl;
        StatusCode st = net->getStats(&pstatsTarget, nullptr);
        if (st == StatusCode::OK && pstatsTarget) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          if (st == StatusCode::OK && pstatsTarget) {" << std::endl;
            pstatsTarget->setNodesStats(networkStats->getNodesStats());
        }
    }

    return net;
}

struct NodePrinter {
    enum FILL_COLOR { DATA, SUPPORTED_LAYER, UNSOPPORTED_LAYER };

    std::unordered_set<InferenceEngine::Data*> printed_data;
    std::unordered_set<InferenceEngine::CNNLayer*> printed_layers;
    std::ostream& out;

    printer_callback layer_cb;

    explicit NodePrinter(std::ostream& os, printer_callback cb): out(os), layer_cb(std::move(cb)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      explicit NodePrinter(std::ostream& os, printer_callback cb): out(os), layer_cb(std::move(cb)) {" << std::endl;}

    bool isPrinted(const CNNLayerPtr& layer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      bool isPrinted(const CNNLayerPtr& layer) {" << std::endl;
        return static_cast<bool>(printed_layers.count(layer.get()));
    }

    bool isPrinted(const DataPtr& datum) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      bool isPrinted(const DataPtr& datum) {" << std::endl;
        return static_cast<bool>(printed_data.count(datum.get()));
    }

    string colorToStr(FILL_COLOR color) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      string colorToStr(FILL_COLOR color) {" << std::endl;
        switch (color) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          switch (color) {" << std::endl;
        case DATA:
            return "#FCF6E3";
        case SUPPORTED_LAYER:
            return "#D9EAD3";
        case UNSOPPORTED_LAYER:
            return "#F4CCCC";
        default:
            return "#FFFFFF";
        }
    }

    string formatSize_(const std::vector<unsigned int>& spatialDims) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      string formatSize_(const std::vector<unsigned int>& spatialDims) {" << std::endl;
        string result;
        if (spatialDims.empty()) return result;
        result = std::to_string(spatialDims[0]);
        for (auto dim : spatialDims) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          for (auto dim : spatialDims) {" << std::endl;
            result += "x" + std::to_string(dim);
        }
        return result;
    }

    string cleanNodeName_(string node_name) const {
        // remove dot and dash symbols from node name. It is incorrectly displayed in xdot
        node_name.erase(remove(node_name.begin(), node_name.end(), '.'), node_name.end());
        std::replace(node_name.begin(), node_name.end(), '-', '_');
        std::replace(node_name.begin(), node_name.end(), ':', '_');
        return node_name;
    }

    void printLayerNode(const CNNLayerPtr& layer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      void printLayerNode(const CNNLayerPtr& layer) {" << std::endl;
        auto node_name = "layer_" + cleanNodeName_(layer->name);
        printed_layers.insert(layer.get());

        ordered_properties printed_properties;

        ordered_properties node_properties = {{"shape", "box"},
                                              {"style", "filled"},
                                              {"fillcolor", colorToStr(SUPPORTED_LAYER)}};

        auto type = layer->type;
        printed_properties.emplace_back("type", type);

        if (type == "Convolution") {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          if (type == 'Convolution') {" << std::endl;
            auto* conv = dynamic_cast<ConvolutionLayer*>(layer.get());

            if (conv != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              if (conv != nullptr) {" << std::endl;
                unsigned int depth = conv->_out_depth, group = conv->_group;

                printed_properties.emplace_back(
                    "kernel size", formatSize_({&(conv->_kernel[0]), &(conv->_kernel[conv->_kernel.size() - 1])}));
                printed_properties.emplace_back("output depth", std::to_string(depth));
                printed_properties.emplace_back("group", std::to_string(group));
                printed_properties.emplace_back(
                    "padding begin", formatSize_({&(conv->_padding[0]), &(conv->_padding[conv->_padding.size() - 1])}));
                printed_properties.emplace_back(
                    "padding end",
                    formatSize_({&(conv->_pads_end[0]), &(conv->_pads_end[conv->_pads_end.size() - 1])}));
                printed_properties.emplace_back(
                    "strides", formatSize_({&(conv->_stride[0]), &(conv->_stride[conv->_stride.size() - 1])}));
                printed_properties.emplace_back(
                    "dilations", formatSize_({&(conv->_dilation[0]), &(conv->_dilation[conv->_dilation.size() - 1])}));
            }
        } else if (type == "Pooling") {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          } else if (type == 'Pooling') {" << std::endl;
            auto* pool = dynamic_cast<PoolingLayer*>(layer.get());

            if (pool != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              if (pool != nullptr) {" << std::endl;
                printed_properties.emplace_back(
                    "window size", formatSize_({&(pool->_kernel[0]), &(pool->_kernel[pool->_kernel.size() - 1])}));
                printed_properties.emplace_back(
                    "padding begin", formatSize_({&(pool->_padding[0]), &(pool->_padding[pool->_padding.size() - 1])}));
                printed_properties.emplace_back(
                    "padding end",
                    formatSize_({&(pool->_pads_end[0]), &(pool->_pads_end[pool->_pads_end.size() - 1])}));
                printed_properties.emplace_back(
                    "strides", formatSize_({&(pool->_stride[0]), &(pool->_stride[pool->_stride.size() - 1])}));
            }
        } else if (type == "ReLU") {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          } else if (type == 'ReLU') {" << std::endl;
            auto* relu = dynamic_cast<ReLULayer*>(layer.get());

            if (relu != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              if (relu != nullptr) {" << std::endl;
                float negative_slope = relu->negative_slope;

                if (negative_slope != 0.0f)
                    printed_properties.emplace_back("negative_slope", std::to_string(negative_slope));
            }
        } else if (type == "Eltwise") {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          } else if (type == 'Eltwise') {" << std::endl;
            auto* eltwise = dynamic_cast<EltwiseLayer*>(layer.get());

            if (eltwise != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              if (eltwise != nullptr) {" << std::endl;
                std::string operation;

                if (eltwise->_operation == EltwiseLayer::Sum)
                    operation = "Sum";
                else if (eltwise->_operation == EltwiseLayer::Prod)
                    operation = "Prod";
                else if (eltwise->_operation == EltwiseLayer::Max)
                    operation = "Max";
                else if (eltwise->_operation == EltwiseLayer::Sub)
                    operation = "Sub";
                else if (eltwise->_operation == EltwiseLayer::Min)
                    operation = "Min";
                else if (eltwise->_operation == EltwiseLayer::Div)
                    operation = "Div";
                else if (eltwise->_operation == EltwiseLayer::Squared_diff)
                    operation = "Squared_diff";
                else if (eltwise->_operation == EltwiseLayer::Equal)
                    operation = "Equal";
                else if (eltwise->_operation == EltwiseLayer::Not_equal)
                    operation = "Not_equal";
                else if (eltwise->_operation == EltwiseLayer::Less)
                    operation = "Less";
                else if (eltwise->_operation == EltwiseLayer::Less_equal)
                    operation = "Less_equal";
                else if (eltwise->_operation == EltwiseLayer::Greater)
                    operation = "Greater";
                else if (eltwise->_operation == EltwiseLayer::Greater_equal)
                    operation = "Greater_equal";
                else if (eltwise->_operation == EltwiseLayer::Logical_NOT)
                    operation = "Logical_NOT";
                else if (eltwise->_operation == EltwiseLayer::Logical_AND)
                    operation = "Logical_AND";
                else if (eltwise->_operation == EltwiseLayer::Logical_OR)
                    operation = "Logical_OR";
                else if (eltwise->_operation == EltwiseLayer::Logical_XOR)
                    operation = "Logical_XOR";
                else if (eltwise->_operation == EltwiseLayer::Floor_mod)
                    operation = "Floor_mod";
                else if (eltwise->_operation == EltwiseLayer::Pow)
                    operation = "Pow";
                else if (eltwise->_operation == EltwiseLayer::Mean)
                    operation = "Mean";

                printed_properties.emplace_back("operation", operation);
            }
        }

        if (layer_cb != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          if (layer_cb != nullptr) {" << std::endl;
            layer_cb(layer, printed_properties, node_properties);
        }

        printNode(node_name, layer->name, node_properties, printed_properties);
    }

    void printDataNode(const std::shared_ptr<Data>& data) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      void printDataNode(const std::shared_ptr<Data>& data) {" << std::endl;
        auto node_name = "data_" + cleanNodeName_(data->getName());
        printed_data.insert(data.get());

        ordered_properties printed_properties;
        ordered_properties node_properties = {{"shape", "ellipse"},
                                              {"style", "filled"},
                                              {"fillcolor", colorToStr(DATA)}};

        std::stringstream dims_ss;
        size_t idx = data->getTensorDesc().getDims().size();
        dims_ss << '[';
        for (auto& dim : data->getTensorDesc().getDims()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          for (auto& dim : data->getTensorDesc().getDims()) {" << std::endl;
            dims_ss << dim << ((--idx) != 0u ? ", " : "");
        }
        dims_ss << ']';

        printed_properties.emplace_back("dims", dims_ss.str());
        printed_properties.emplace_back("precision", data->getPrecision().name());

        std::stringstream ss;
        ss << data->getTensorDesc().getLayout();
        printed_properties.emplace_back("layout", ss.str());
        printed_properties.emplace_back("name", data->getName());
        if (data->getCreatorLayer().lock() != nullptr)
            printed_properties.emplace_back("creator layer", data->getCreatorLayer().lock()->name);
        printNode(node_name, data->getName(), node_properties, printed_properties);
    }

    void printNode(string const& node_name, const string& node_title, ordered_properties const& node_properties,
                   ordered_properties const& printed_properties) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:                     ordered_properties const& printed_properties) {" << std::endl;
        // normalization of names, removing all prohibited symbols like "/"
        string nodeNameN = node_name;
        std::replace(nodeNameN.begin(), nodeNameN.end(), '/', '_');
        string dataNameN = node_title;
        std::replace(dataNameN.begin(), dataNameN.end(), '/', '_');

        out << '\t' << nodeNameN << " [";
        for (auto& node_property : node_properties) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          for (auto& node_property : node_properties) {" << std::endl;
            out << node_property.first << "=\"" << node_property.second << "\", ";
        }

        out << "label=\"" << node_title;
        for (auto& printed_property : printed_properties) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          for (auto& printed_property : printed_properties) {" << std::endl;
            out << "\\n" << printed_property.first << ": " << printed_property.second;
        }
        out << "\"];\n";
    }

    void printEdge(const CNNLayerPtr& from_, const DataPtr& to_, bool reverse) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      void printEdge(const CNNLayerPtr& from_, const DataPtr& to_, bool reverse) {" << std::endl;
        auto from_name = "layer_" + cleanNodeName_(from_->name);
        auto to_name = "data_" + cleanNodeName_(to_->getName());
        std::replace(from_name.begin(), from_name.end(), '/', '_');
        std::replace(to_name.begin(), to_name.end(), '/', '_');
        if (reverse) std::swap(from_name, to_name);
        out << '\t' << from_name << " -> " << to_name << ";\n";
    }
};

void saveGraphToDot(InferenceEngine::ICNNNetwork& network, std::ostream& out, printer_callback layer_cb) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  void saveGraphToDot(InferenceEngine::ICNNNetwork& network, std::ostream& out, printer_callback layer_cb) {" << std::endl;
    NodePrinter printer(out, std::move(layer_cb));

    out << "digraph Network {\n";
    // Traverse graph and print nodes
    for (const auto& layer : details::CNNNetSortTopologically(network)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      for (const auto& layer : details::CNNNetSortTopologically(network)) {" << std::endl;
        printer.printLayerNode(layer);

        // Print output Data Object
        for (auto& dataptr : layer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          for (auto& dataptr : layer->outData) {" << std::endl;
            if (!printer.isPrinted(dataptr)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              if (!printer.isPrinted(dataptr)) {" << std::endl;
                printer.printDataNode(dataptr);
            }
            printer.printEdge(layer, dataptr, false);
        }

        // Print input Data objects
        for (auto& datum : layer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:          for (auto& datum : layer->insData) {" << std::endl;
            auto dataptr = datum.lock();
            if (!printer.isPrinted(dataptr)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              if (!printer.isPrinted(dataptr)) {" << std::endl;
                printer.printDataNode(dataptr);
            }
            printer.printEdge(layer, dataptr, true);
        }
    }
    out << "}" << std::endl;
}

std::unordered_set<DataPtr> getRootDataObjects(ICNNNetwork& network) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  std::unordered_set<DataPtr> getRootDataObjects(ICNNNetwork& network) {" << std::endl;
    std::unordered_set<DataPtr> ret;
    details::CNNNetworkIterator i(&network);
    while (i != details::CNNNetworkIterator()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      while (i != details::CNNNetworkIterator()) {" << std::endl;
        CNNLayer::Ptr layer = *i;

        // TODO: Data without creatorLayer
        if (CaselessEq<string>()(layer->type, "input") || CaselessEq<string>()(layer->type, "const") ||
            CaselessEq<string>()(layer->type, "memory")) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:              CaselessEq<string>()(layer->type, 'memory')) {" << std::endl;
            ret.insert(layer->outData.begin(), layer->outData.end());
        }
        i++;
    }
    return ret;
}

namespace {

template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C> >
std::basic_string<C> getPathName(const std::basic_string<C>& s) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  std::basic_string<C> getPathName(const std::basic_string<C>& s) {" << std::endl;
    size_t i = s.rfind(FileUtils::FileTraits<C>::FileSeparator, s.length());
    if (i != std::string::npos) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:      if (i != std::string::npos) {" << std::endl;
        return (s.substr(0, i));
    }

    return {};
}

}  // namespace

#ifndef _WIN32

static std::string getIELibraryPathUnix() {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  static std::string getIELibraryPathUnix() {" << std::endl;
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    return getPathName(std::string(info.dli_fname)).c_str();
}

#endif  // _WIN32

#ifdef ENABLE_UNICODE_PATH_SUPPORT

std::wstring getIELibraryPathW() {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  std::wstring getIELibraryPathW() {" << std::endl;
#if defined(_WIN32) || defined(_WIN64)
    wchar_t ie_library_path[4096];
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)getIELibraryPath, &hm)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:                              (LPCWSTR)getIELibraryPath, &hm)) {" << std::endl;
        THROW_IE_EXCEPTION << "GetModuleHandle returned " << GetLastError();
    }
    GetModuleFileNameW(hm, (LPWSTR)ie_library_path, sizeof(ie_library_path));
    return getPathName(std::wstring(ie_library_path));
#else
    Dl_info info;
    std::cerr << "dldt ie_util_internal.cpp getIELibraryPathW: before dladdr" << std::endl;
    dladdr(reinterpret_cast<void*>(getIELibraryPath), &info);
    std::cerr << "dldt ie_util_internal.cpp getIELibraryPathW: after dladdr" << std::endl;
    return details::multiByteCharToWString(getIELibraryPathUnix().c_str());
#endif
}

#endif

std::string getIELibraryPath() {
    std::cerr << "./inference-engine/src/inference_engine/ie_util_internal.cpp:  std::string getIELibraryPath() {" << std::endl;
#ifdef ENABLE_UNICODE_PATH_SUPPORT
    return details::wStringtoMBCSstringChar(getIELibraryPathW());
#else
    return getIELibraryPathUnix();
#endif
}

}  // namespace InferenceEngine
