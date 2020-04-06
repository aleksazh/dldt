#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer.h"

#include <cpp/ie_cnn_network.h>
#include <details/ie_cnn_network_tools.h>

#include <details/caseless.hpp>
#include <iterator>
#include <map>
#include <utility>
#include <memory>
#include <shape_infer/const_infer/ie_const_infer_holder.hpp>
#include <string>
#include <vector>
#include <mutex>

#include "blob_factory.hpp"
#include "cnn_network_impl.hpp"
#include "graph_tools.hpp"
#include "net_pass.h"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace InferenceEngine {

bool isForFakeQuantzie(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:  bool isForFakeQuantzie(const CNNLayer& layer) {" << std::endl;
    for (const DataPtr data : layer.outData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      for (const DataPtr data : layer.outData) {" << std::endl;
        for (const auto it : data->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          for (const auto it : data->getInputTo()) {" << std::endl;
            const CNNLayerPtr childLayer = it.second;
            if (childLayer->type == "FakeQuantize" || childLayer->type == "Quantize") {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              if (childLayer->type == 'FakeQuantize' || childLayer->type == 'Quantize') {" << std::endl;
                return true;
            }
        }
    }

    return false;
}

static std::vector<DataPtr> get_inputs(details::CNNNetworkImpl* _network) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:  static std::vector<DataPtr> get_inputs(details::CNNNetworkImpl* _network) {" << std::endl;
    if (!_network) return {};

    InputsDataMap ins_info;
    _network->getInputsInfo(ins_info);

    std::vector<DataPtr> inputs;
    for (const auto& kvp : ins_info)
        inputs.push_back(kvp.second->getInputData());
    return inputs;
}

static std::vector<DataPtr> get_outputs(details::CNNNetworkImpl* _network) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:  static std::vector<DataPtr> get_outputs(details::CNNNetworkImpl* _network) {" << std::endl;
    if (!_network) return {};

    std::map<std::string, DataPtr> outs_info;
    _network->getOutputsInfo(outs_info);

    std::vector<DataPtr> outputs;
    for (const auto& kvp : outs_info)
        outputs.push_back(kvp.second);
    return outputs;
}

ConstTransformer::ConstTransformer(details::CNNNetworkImpl* _network)
        : inputs(get_inputs(_network)), outputs(get_outputs(_network)), network(_network) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          : inputs(get_inputs(_network)), outputs(get_outputs(_network)), network(_network) {" << std::endl;
    if (!_network)
        THROW_IE_EXCEPTION << "[ERROR]: Failed to init ConstTransformer with null pointer of network";
}

ConstTransformer::ConstTransformer(std::vector<DataPtr> &_inputs, std::vector<DataPtr> &_outputs)
        : inputs(_inputs), outputs(_outputs), network(nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          : inputs(_inputs), outputs(_outputs), network(nullptr) {" << std::endl;
    if (inputs.empty() || outputs.empty())
        THROW_IE_EXCEPTION << "[ERROR]: Failed to init ConstTransformer with empty list of inputs or outputs";
}

std::vector<CNNLayerPtr> ConstTransformer::foldConstSubgraphsInternal(const std::map<std::string, bool>& constLayers,
                                                                      const BlobMap& constData,
                                                                      const std::vector<CNNLayerPtr>& sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                                                                        const std::vector<CNNLayerPtr>& sortedLayers) {" << std::endl;
    std::vector<CNNLayerPtr> remainingConstLayers;
    for (const auto& layer : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      for (const auto& layer : sortedLayers) {" << std::endl;
        if (constLayers.find(layer->name) != constLayers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          if (constLayers.find(layer->name) != constLayers.end()) {" << std::endl;
            // const layer doesn't need parent connections -> erase them
            for (const auto& insData : layer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              for (const auto& insData : layer->insData) {" << std::endl;
                auto& inputTo = insData.lock()->getInputTo();
                inputTo.erase(layer->name);
                // Note: to resolve corner case above layers can be marked as const with const data, just to be removed
                // properly.. and maybe this logic wouldn't be needed
                if (inputTo.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  if (inputTo.empty()) {" << std::endl;
                    auto creator = insData.lock()->getCreatorLayer().lock();
                    auto it = std::find(creator->outData.begin(), creator->outData.end(), insData.lock());
                    if (it != creator->outData.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      if (it != creator->outData.end()) {" << std::endl;
                        data_to_remove.push_back(*it);
                        creator->outData.erase(it);
                    }
                }
            }
            layer->insData.clear();

            if (constLayers.at(layer->name)) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              if (constLayers.at(layer->name)) {" << std::endl;
                for (const auto& outData : layer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  for (const auto& outData : layer->outData) {" << std::endl;
                    for (const auto& inputTo : outData->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      for (const auto& inputTo : outData->getInputTo()) {" << std::endl;
                        CNNLayerPtr inputToLayer;
                        std::string inputToName;
                        std::tie(inputToName, inputToLayer) = inputTo;
                        auto& insData = inputToLayer->insData;
                        auto insDataIt =
                            std::find_if(insData.begin(), insData.end(), [&outData](const DataWeakPtr& current) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                              std::find_if(insData.begin(), insData.end(), [&outData](const DataWeakPtr& current) {" << std::endl;
                                return current.lock()->getName() == outData->getName();
                            });
                        // remove connection with const data, because for const child it's not needed, for dynamic - new
                        // one will be created
                        if (insDataIt != insData.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                          if (insDataIt != insData.end()) {" << std::endl;
                            insDataIt = inputToLayer->insData.erase(insDataIt);
                        }
                    }
                    data_to_remove.push_back(outData);
                }
                layer_to_remove.push_back(layer);
            } else {
                // if only one output data is not const - do nothing, otherwise - run procedure below
                // note: multiple const output data requires multiple layers with blob["custom"] to keep const data
                bool keepConstData = layer->outData.size() == 1;
                if (keepConstData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  if (keepConstData) {" << std::endl;
                    auto outData = layer->outData[0];
                    for (const auto& inputTo : outData->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      for (const auto& inputTo : outData->getInputTo()) {" << std::endl;
                        if (constLayers.find(inputTo.first) != constLayers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                          if (constLayers.find(inputTo.first) != constLayers.end()) {" << std::endl;
                            keepConstData = false;
                        }
                    }
                }
                if (keepConstData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  if (keepConstData) {" << std::endl;
                    if (!constLayers.at(layer->name)) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      if (!constLayers.at(layer->name)) {" << std::endl;
                        auto outData = layer->outData[0];
                        if (layer->blobs.find("custom") == layer->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                          if (layer->blobs.find('custom') == layer->blobs.end()) {" << std::endl;
                            // if there's no const data - set it
                            const auto it = constData.find(outData->getName());
                            if (it != constData.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                              if (it != constData.end()) {" << std::endl;
                                layer->blobs["custom"] = it->second;
                            }
                        }
                        if (layer->type != "Const") {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                          if (layer->type != 'Const') {" << std::endl;
                            // layer was calculated during the Const Propagation, need to hide its semantic (type,
                            // params)
                            LayerParams layerParams {layer->name + "__" + outData->getName() + "__Const", "Const",
                                                     layer->precision};
                            auto newLayer = std::make_shared<CNNLayer>(layerParams);
                            for (const auto& data : layer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                              for (const auto& data : layer->outData) {" << std::endl;
                                data->getCreatorLayer() = newLayer;
                            }
                            newLayer->outData = layer->outData;
                            newLayer->blobs["custom"] = layer->blobs["custom"];
                            layer_to_remove.push_back(layer);
                            layer_to_add.push_back(newLayer);
                            remainingConstLayers.push_back(newLayer);
                        } else {
                            // Layer with `Const` type should be also considered on trimming shape inputs
                            remainingConstLayers.push_back(layer);
                        }
                    }
                } else {
                    for (const auto& outData : layer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      for (const auto& outData : layer->outData) {" << std::endl;
                        for (const auto& inputTo : outData->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                          for (const auto& inputTo : outData->getInputTo()) {" << std::endl;
                            CNNLayerPtr inputToLayer;
                            std::string inputToName;
                            std::tie(inputToName, inputToLayer) = inputTo;
                            auto& insData = inputToLayer->insData;
                            auto insDataIt =
                                std::find_if(insData.begin(), insData.end(), [&outData](const DataWeakPtr& current) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                                  std::find_if(insData.begin(), insData.end(), [&outData](const DataWeakPtr& current) {" << std::endl;
                                    return current.lock()->getName() == outData->getName();
                                });
                            // remove connection with const data, because for const child it's not needed, for dynamic -
                            // new one will be created
                            if (insDataIt != insData.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                              if (insDataIt != insData.end()) {" << std::endl;
                                insDataIt = inputToLayer->insData.erase(insDataIt);
                            }
                            if (constLayers.find(inputToName) == constLayers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                              if (constLayers.find(inputToName) == constLayers.end()) {" << std::endl;
                                // next layer is not const, need to attach const data to it via blobs["custom"] of new
                                // Const layer
                                LayerParams layerParams {layer->name + "__" + outData->getName() + "__Const", "Const",
                                                         layer->precision};
                                auto newLayer = std::make_shared<CNNLayer>(layerParams);
                                remainingConstLayers.push_back(newLayer);
                                const auto it = constData.find(outData->getName());
                                if (it != constData.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                                  if (it != constData.end()) {" << std::endl;
                                    newLayer->blobs["custom"] = it->second;
                                }
                                auto newData = std::make_shared<Data>(outData->getName() + "__" + inputToName,
                                                                      outData->getTensorDesc());
                                newData->getCreatorLayer() = newLayer;
                                newData->getInputTo()[inputToName] = inputToLayer;
                                newLayer->outData = {newData};
                                layer_to_add.push_back(newLayer);
                                data_to_add.push_back(newData);
                                inputToLayer->insData.insert(insDataIt, newData);
                            }
                        }
                    }
                    for (const auto& data : layer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      for (const auto& data : layer->outData) {" << std::endl;
                        data_to_remove.push_back(data);
                    }
                    layer_to_remove.push_back(layer);
                }
            }
        }
        if (NetPass::HasInternalSubnet(layer)) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          if (NetPass::HasInternalSubnet(layer)) {" << std::endl;
            auto subgraph = NetPass::GetInternalSubnet(layer);
            ConstTransformer transformer(subgraph.inputs, subgraph.outputs);
            transformer.foldConstSubgraphs();
        }
    }
    return remainingConstLayers;
}

const std::map<std::string, bool> ConstTransformer::getConstLayers(const std::vector<CNNLayerPtr>& sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:  const std::map<std::string, bool> ConstTransformer::getConstLayers(const std::vector<CNNLayerPtr>& sortedLayers) {" << std::endl;
    std::map<std::string, bool> mapConstLayers;
    // collect all const layers, which inputs are const layers.
    for (const auto& layer : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      for (const auto& layer : sortedLayers) {" << std::endl;
        // Layers with "Shape" and "Const" type are Const by definition
        if (layer->type == "Shape" || layer->type == "Const") {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          if (layer->type == 'Shape' || layer->type == 'Const') {" << std::endl;
            mapConstLayers[layer->name] = false;
        } else if ((layer->type != "FakeQuantize") && (layer->type != "Quantize") && (!isForFakeQuantzie(*layer))) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          } else if ((layer->type != 'FakeQuantize') && (layer->type != 'Quantize') && (!isForFakeQuantzie(*layer))) {" << std::endl;
            bool isAllInputsConst = true;
            for (auto const& data : layer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              for (auto const& data : layer->insData) {" << std::endl;
                auto creator = data.lock()->getCreatorLayer().lock();
                if (creator != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  if (creator != nullptr) {" << std::endl;
                    if (mapConstLayers.find(creator->name) == mapConstLayers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      if (mapConstLayers.find(creator->name) == mapConstLayers.end()) {" << std::endl;
                        isAllInputsConst = false;
                    }
                } else {
                    // Empty creator means that it's a network representation via inputs/outs data collection
                    // And it's a firs layer in network.
                    isAllInputsConst = false;
                }
            }
            if (isAllInputsConst && !layer->insData.empty()) mapConstLayers[layer->name] = false;
        }
    }
    // Add mark for const layers, if it's used for shape taking layers as second input
    // true - is used and can be deleted from graph, as no influence on data, false - opposite
    std::map<std::string, bool> mapVisitedLayers = mapConstLayers;
    for (auto rit = sortedLayers.rbegin(); rit != sortedLayers.rend(); rit++) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      for (auto rit = sortedLayers.rbegin(); rit != sortedLayers.rend(); rit++) {" << std::endl;
        auto currentLayer = (*rit);
        std::string currentLayerName = currentLayer->name;
        bool isCurrentConst = mapConstLayers.find(currentLayerName) != mapConstLayers.end();
        for (int i = 0; i < currentLayer->insData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          for (int i = 0; i < currentLayer->insData.size(); i++) {" << std::endl;
            std::string creatorName;
            if (currentLayer->insData[i].lock() != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              if (currentLayer->insData[i].lock() != nullptr) {" << std::endl;
                auto creator = currentLayer->insData[i].lock()->getCreatorLayer().lock();
                if (creator) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  if (creator) {" << std::endl;
                    creatorName = creator->name;
                }
            }
            bool isCreatorConst = mapConstLayers.find(creatorName) != mapConstLayers.end();
            if (isCreatorConst) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              if (isCreatorConst) {" << std::endl;
                // mark second const input of shape taking layers (Reshape, Interp..), if they wasn't visited before
                if ((i == 1) && (shapeTaking.find(currentLayer->type)) != shapeTaking.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  if ((i == 1) && (shapeTaking.find(currentLayer->type)) != shapeTaking.end()) {" << std::endl;
                    if (!mapConstLayers[creatorName]) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      if (!mapConstLayers[creatorName]) {" << std::endl;
                        if (!mapVisitedLayers.at(creatorName)) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                          if (!mapVisitedLayers.at(creatorName)) {" << std::endl;
                            mapConstLayers[creatorName] = true;
                        }
                    }
                } else {
                    if (isCurrentConst) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      if (isCurrentConst) {" << std::endl;
                        if (mapConstLayers.at(currentLayerName)) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                          if (mapConstLayers.at(currentLayerName)) {" << std::endl;
                            if (!mapConstLayers[creatorName]) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                              if (!mapConstLayers[creatorName]) {" << std::endl;
                                if (!mapVisitedLayers.at(creatorName)) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                                  if (!mapVisitedLayers.at(creatorName)) {" << std::endl;
                                    mapConstLayers[creatorName] = true;
                                }
                            }
                        } else {
                            mapConstLayers[creatorName] = false;
                        }
                    } else {
                        mapConstLayers[creatorName] = false;
                    }
                }
            }
            mapVisitedLayers[creatorName] = true;
        }
        mapVisitedLayers[currentLayerName] = true;
    }
    return mapConstLayers;
}

const BlobMap ConstTransformer::getConstData(const std::map<std::string, bool>& constLayers,
                                             const std::vector<CNNLayerPtr>& sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                                               const std::vector<CNNLayerPtr>& sortedLayers) {" << std::endl;
    ShapeInfer::ConstInferHolder holder;
    BlobMap constData;
    auto getInputBlobs = [&constData](const std::vector<DataWeakPtr>& insData,
                                      bool isForShape) -> std::vector<Blob::CPtr> {
        std::vector<Blob::CPtr> inputBlobs;
        // special case of Const layers: no inputs, no input blobs
        if (insData.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          if (insData.empty()) {" << std::endl;
            return {};
        }
        for (const auto& data : insData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          for (const auto& data : insData) {" << std::endl;
            std::string dataName = data.lock()->getName();
            if (constData.find(dataName) != constData.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              if (constData.find(dataName) != constData.end()) {" << std::endl;
                // get blobs, inferred before
                inputBlobs.push_back(constData.at(dataName));
            } else {
                // special case of Shape layer: no input data, but blob contains info about dimensions, layout and
                // etc...
                auto blob = make_blob_with_precision(data.lock()->getTensorDesc());
                inputBlobs.push_back(blob);
            }
        }
        return inputBlobs;
    };

    auto getOutputBlobs = [](const std::vector<DataPtr>& outData) -> std::vector<Blob::Ptr> {
        std::vector<Blob::Ptr> outputBlobs;
        for (const auto& data : outData) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          for (const auto& data : outData) {" << std::endl;
            auto blob = make_blob_with_precision(data->getTensorDesc());
            blob->allocate();
            outputBlobs.push_back(blob);
        }
        return outputBlobs;
    };

    for (const auto& layer : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      for (const auto& layer : sortedLayers) {" << std::endl;
        if (layer->type == "FakeQuantize" || layer->type == "Quantize") {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          if (layer->type == 'FakeQuantize' || layer->type == 'Quantize') {" << std::endl;
            continue;
        }

        if (constLayers.find(layer->name) != constLayers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          if (constLayers.find(layer->name) != constLayers.end()) {" << std::endl;
            std::string layerName = layer->name;
            bool isForShape = constLayers.at(layerName);

            auto implPtr = holder.getConstInferImpl(layer->type);
            if (!implPtr && !isForShape)
                if (layer->type != "FakeQuantize" && layer->type != "Quantize")
                    THROW_IE_EXCEPTION << "Failed to find reference implementation for `" + layer->name +
                                              "` Layer with `" + layer->type + "` Type on constant propagation";
            if (!isForShape) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              if (!isForShape) {" << std::endl;
                auto outputBlobs = getOutputBlobs(layer->outData);
                auto inp = getInputBlobs(layer->insData, isForShape);
                if (layer->type != "FakeQuantize" && layer->type != "Quantize")
                    implPtr->infer(inp, layer->params, layer->blobs, outputBlobs);
                for (int i = 0; i < layer->outData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  for (int i = 0; i < layer->outData.size(); i++) {" << std::endl;
                    std::string dataName = layer->outData[i]->getName();
                    auto shapes = layer->outData[i]->getTensorDesc().getDims();
                    outputBlobs[i]->getTensorDesc().reshape(shapes, TensorDesc::getLayoutByDims(shapes));
                    constData[dataName] = outputBlobs[i];
                }
            }
        }
    }
    return constData;
}

/**
 * Will replace provided layer with reshape with corresponding shape from output data
 *
 * @param layer is operation to replace with static reshape
 * @return newly created reshape static layer
 */
static CNNLayerPtr replace_with_static_reshape(CNNLayerPtr &layer) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:  static CNNLayerPtr replace_with_static_reshape(CNNLayerPtr &layer) {" << std::endl;
    IE_ASSERT(layer->insData.size() == 1);
    IE_ASSERT(layer->outData.size() == 1);

    auto in_data = layer->insData[0].lock();
    if (in_data == nullptr)
        THROW_IE_EXCEPTION << "Layer '" << layer->name << "' has invalid input data";
    auto out_data = layer->outData[0];

    auto precision = out_data->getPrecision();
    auto shape = out_data->getDims();

    // TODO: Have to use old name instead a new one because tensor statistic is mapped
    //       to layers by name. The old int8 pipeline may be broken because of lose
    //       tensor statistic for particular reshape.
    auto reshape = std::make_shared<ReshapeLayer>(
            LayerParams{layer->name, "Reshape", precision});
    reshape->shape = std::vector<int>(shape.begin(), shape.end());

    // replacement
    auto &input_to_map = in_data->getInputTo();

    // try to find by name
    auto found_by_name = input_to_map.find(layer->name);
    if (found_by_name != input_to_map.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      if (found_by_name != input_to_map.end()) {" << std::endl;
        input_to_map.erase(found_by_name);
    } else {
        // try to find by ptr
        auto found_by_ptr = std::find_if(input_to_map.begin(), input_to_map.end(),
                                         [&layer] (const std::pair<std::string, CNNLayerPtr> &p)
                                         {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                                           [&layer] (const std::pair<std::string, CNNLayerPtr> &p)                                          {" << std::endl; return p.second == layer; });
        if (found_by_ptr != input_to_map.end())
            input_to_map.erase(found_by_ptr);
    }
    input_to_map[reshape->name] = reshape;

    reshape->insData = {in_data};
    reshape->outData = {out_data};
    out_data->getCreatorLayer() = reshape;

    return reshape;
}

void ConstTransformer::trimShapeInputs(const std::vector<CNNLayerPtr>& constLayers,
                                       std::vector<CNNLayerPtr>& allLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                                         std::vector<CNNLayerPtr>& allLayers) {" << std::endl;
    for (const auto& layer : constLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      for (const auto& layer : constLayers) {" << std::endl;
        if (layer->outData.size() == 1 && layer->type == "Const" && layer->insData.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          if (layer->outData.size() == 1 && layer->type == 'Const' && layer->insData.empty()) {" << std::endl;
            auto constData = layer->outData[0];
            std::map<std::string, CNNLayerPtr> inputToMap = constData->getInputTo();
            for (const auto& inputTo : inputToMap) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              for (const auto& inputTo : inputToMap) {" << std::endl;
                CNNLayerPtr inputToLayer = inputTo.second;
                if (shapeTaking.find(inputToLayer->type) != shapeTaking.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  if (shapeTaking.find(inputToLayer->type) != shapeTaking.end()) {" << std::endl;
                    auto& insData = inputToLayer->insData;
                    auto it = std::find_if(insData.begin(), insData.end(), [&constData](const DataWeakPtr& current) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      auto it = std::find_if(insData.begin(), insData.end(), [&constData](const DataWeakPtr& current) {" << std::endl;
                        return current.lock()->getName() == constData->getName();
                    });
                    if (it != insData.end() && std::distance(insData.begin(), it) == 1) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                      if (it != insData.end() && std::distance(insData.begin(), it) == 1) {" << std::endl;
                        inputToLayer->insData.erase(it);
                        constData->getInputTo().erase(inputTo.first);
                    }
                }
            }
            if (constData->getInputTo().empty()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              if (constData->getInputTo().empty()) {" << std::endl;
                layer_to_remove.push_back(layer);
                data_to_remove.push_back(constData);
            }
        }
    }
    // TODO: Some WA. Previous step foldConstSubgraphsInternal remove all const data
    //       from graph. Although that is responsibility of trimShapeInputs pass.
    //       That's why we need make additional pass through allLayers and replace
    //       all shape taken layers like Squeeze/Flatten with Reshape with single input.
    for (auto& layer : allLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      for (auto& layer : allLayers) {" << std::endl;
        // Layer is from list of reshape-like layers
        if (layer->type != "Reshape" &&
            layer->type != "Unsqueeze" &&
            layer->type != "Squeeze" &&
            layer->type != "Flatten")
            continue;

        // already removed
        if (std::find(layer_to_remove.begin(), layer_to_remove.end(), layer) != layer_to_remove.end())
            continue;

        // The second input was not removed. So shape is not constant.
        if (layer->insData.size() != 1)
            continue;

        auto new_one = replace_with_static_reshape(layer);
        layer_to_remove.push_back(layer);
        layer_to_add.push_back(new_one);
    }
}

void ConstTransformer::cleanup() {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:  void ConstTransformer::cleanup() {" << std::endl;
    if (network) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      if (network) {" << std::endl;
        for (const auto &layer : layer_to_remove) network->removeLayer(layer->name);
        for (const auto &data : data_to_remove) network->removeData(data->getName());

        for (const auto &layer : layer_to_add) network->addLayer(layer);
        for (const auto &data : data_to_add) network->addData(data->getName().c_str(), data);
    } else {
        // Subgraph case
        auto &const_holder = inputs.back();
        if (const_holder->getPrecision() == Precision::UNSPECIFIED) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          if (const_holder->getPrecision() == Precision::UNSPECIFIED) {" << std::endl;
            auto &holder_map = const_holder->getInputTo();
            // Remove from const holder data object
            for (const auto &layer : layer_to_remove) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              for (const auto &layer : layer_to_remove) {" << std::endl;
                auto self_found = std::find_if(holder_map.begin(), holder_map.end(),
                        [&layer] (const std::pair<std::string, CNNLayerPtr> kvp) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                          [&layer] (const std::pair<std::string, CNNLayerPtr> kvp) {" << std::endl;
                    return kvp.second == layer;
                });

                if (self_found != holder_map.end()) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:                  if (self_found != holder_map.end()) {" << std::endl;
                    holder_map.erase(self_found);
                }
            }
            // Add to const holder
            for (const auto &layer : layer_to_add) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:              for (const auto &layer : layer_to_add) {" << std::endl;
                holder_map[layer->name] = layer;
            }
        }
    }
}

void ConstTransformer::foldConstSubgraphs() {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:  void ConstTransformer::foldConstSubgraphs() {" << std::endl;
    auto sortedLayers = details::CNNSubnetSortTopologically({inputs, outputs});
    auto constLayers = getConstLayers(sortedLayers);
    auto constData = getConstData(constLayers, sortedLayers);
    foldConstSubgraphsInternal(constLayers, constData, sortedLayers);

    cleanup();
}

void ConstTransformer::fullTrim() {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:  void ConstTransformer::fullTrim() {" << std::endl;
    // Avoid data races on one network instance
    static std::mutex lockFullTrim;
    std::lock_guard<std::mutex> lock(lockFullTrim);
    auto sortedLayers = details::CNNSubnetSortTopologically({inputs, outputs});
    auto constMapLayers = getConstLayers(sortedLayers);
    auto constData = getConstData(constMapLayers, sortedLayers);
    auto constLayers = foldConstSubgraphsInternal(constMapLayers, constData, sortedLayers);
    trimShapeInputs(constLayers, sortedLayers);

    for (auto &layer : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:      for (auto &layer : sortedLayers) {" << std::endl;
        if (NetPass::HasInternalSubnet(layer)) {
    std::cerr << "./inference-engine/src/inference_engine/graph_transformer.cpp:          if (NetPass::HasInternalSubnet(layer)) {" << std::endl;
            auto subgraph = NetPass::GetInternalSubnet(layer);

            ConstTransformer transformer(subgraph.inputs, subgraph.outputs);
            auto ti_sortedLayers = details::CNNSubnetSortTopologically({subgraph.inputs, subgraph.outputs});
            auto ti_constMapLayers = transformer.getConstLayers(ti_sortedLayers);
            auto ti_constData = transformer.getConstData(ti_constMapLayers, ti_sortedLayers);
            auto ti_constLayers = transformer.foldConstSubgraphsInternal(ti_constMapLayers, ti_constData, ti_sortedLayers);
            transformer.trimShapeInputs(ti_constLayers, ti_sortedLayers);
            transformer.cleanup();
        }
    }

    cleanup();
}
}  // namespace InferenceEngine
