#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_const_layer.hpp>
#include <builders/ie_input_layer.hpp>
#include <builders/ie_network_builder.hpp>
#include <details/caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <shape_infer/ie_reshaper.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "blob_factory.hpp"
#include "graph_tools.hpp"
#include "ie_cnn_layer_builder.h"
#include "ie_profiling.hpp"

using namespace InferenceEngine;

/******************************************************************************
 Network builder
 ******************************************************************************/
Builder::Network::Network(const std::string& name): Builder::Network(Context(), name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Network::Network(const std::string& name): Builder::Network(Context(), name) {" << std::endl;}
Builder::Network::Network(const INetwork& network): Builder::Network(Context(), network) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Network::Network(const INetwork& network): Builder::Network(Context(), network) {" << std::endl;}
Builder::Network::Network(const ICNNNetwork& network): Builder::Network(Context(), network) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Network::Network(const ICNNNetwork& network): Builder::Network(Context(), network) {" << std::endl;}

Builder::Network::Network(const Context& ieContext, const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Network::Network(const Context& ieContext, const std::string& name) {" << std::endl;
    parameters["name"] = name;
    parameters["context"] = ieContext;
    parameters["version"] = 3;
    parameters["layers"] = std::vector<Layer::Ptr>();
    parameters["connections"] = std::vector<Connection>();
}

Builder::Network::Network(const Context& ieContext, const INetwork& network): Network(ieContext, network.getName()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Network::Network(const Context& ieContext, const INetwork& network): Network(ieContext, network.getName()) {" << std::endl;
    for (const auto& layer : network) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (const auto& layer : network) {" << std::endl;
        parameters["layers"].as<std::vector<Layer::Ptr>>().push_back(std::make_shared<Layer>(layer));
        const auto layerConnections = network.getLayerConnections(layer->getId());
        for (const auto& connection : layerConnections) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& connection : layerConnections) {" << std::endl;
            bool found = false;
            for (const auto& con : parameters["connections"].as<std::vector<Connection>>()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (const auto& con : parameters['connections'].as<std::vector<Connection>>()) {" << std::endl;
                if (con == connection) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                  if (con == connection) {" << std::endl;
                    found = true;
                    break;
                }
            }
            if (!found) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (!found) {" << std::endl;
                parameters["connections"].as<std::vector<Connection>>().push_back(connection);
            }
        }
    }
}

Builder::Network::Network(const Context& ieContext, const ICNNNetwork& network): Network(ieContext, network.getName()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Network::Network(const Context& ieContext, const ICNNNetwork& network): Network(ieContext, network.getName()) {" << std::endl;
    parameters["version"] = 0;
    auto allInputs = CNNNetGetAllInputLayers(network);
    InputsDataMap inputs;
    network.getInputsInfo(inputs);
    if (inputs.empty() && allInputs.empty())
        THROW_IE_EXCEPTION << "Cannot create graph! No inputs for the topology " << network.getName();

    std::unordered_map<std::string, idx_t> name2id;
    std::unordered_set<Data*> dataPtrs;
    std::vector<CNNLayerPtr> queueLayers;

    auto createGenericFromCNNLayer = [&](const CNNLayerPtr& cnnLayer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      auto createGenericFromCNNLayer = [&](const CNNLayerPtr& cnnLayer) {" << std::endl;
        for (const auto& data : cnnLayer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& data : cnnLayer->insData) {" << std::endl;
            auto lockedData = data.lock();
            if (!lockedData) continue;
            if (dataPtrs.find(lockedData.get()) == dataPtrs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (dataPtrs.find(lockedData.get()) == dataPtrs.end()) {" << std::endl;
                dataPtrs.insert(lockedData.get());
            }
        }
        for (const auto& data : cnnLayer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& data : cnnLayer->outData) {" << std::endl;
            if (dataPtrs.find(data.get()) == dataPtrs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (dataPtrs.find(data.get()) == dataPtrs.end()) {" << std::endl;
                dataPtrs.insert(data.get());
            }
        }
        std::map<std::string, Blob::Ptr> blobs = cnnLayer->blobs;
        size_t inputsCount(0);
        for (const auto& data : cnnLayer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& data : cnnLayer->insData) {" << std::endl;
            auto lockedData = data.lock();
            if (!lockedData) continue;
            inputsCount++;
        }
        const auto layer = builderFromCNNLayer(cnnLayer);
        idx_t layerId = addLayer(layer);

        if (blobs.find("weights") != blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (blobs.find('weights') != blobs.end()) {" << std::endl;
            idx_t constLayerId = addLayer(ConstLayer("weights").setData(blobs["weights"]));
            connect({constLayerId}, {layerId, inputsCount++});
        }
        if (blobs.find("biases") != blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (blobs.find('biases') != blobs.end()) {" << std::endl;
            if (blobs.find("weights") == blobs.end()) ++inputsCount;

            idx_t constLayerId = addLayer(ConstLayer("biases").setData(blobs["biases"]));
            connect({constLayerId}, {layerId, inputsCount++});
        }
        for (const auto& it : blobs) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& it : blobs) {" << std::endl;
            if (it.first == "weights" || it.first == "biases") continue;
            idx_t constLayerId = addLayer(ConstLayer(it.first).setData(it.second));
            connect({constLayerId}, {layerId, inputsCount++});
        }
        name2id[layer.getName()] = layerId;
        return layerId;
    };

    auto addPreProcessFor = [&](const InputInfo::Ptr& inputInfo) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      auto addPreProcessFor = [&](const InputInfo::Ptr& inputInfo) {" << std::endl;
        auto inputLayer = getLayer(name2id[inputInfo->name()]);
        if (inputLayer->getType().empty() && inputLayer->getName().empty()) return;

        inputLayer->getParameters()["preProcess"] = inputInfo->getPreProcess();
    };

    for (auto input : inputs) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (auto input : inputs) {" << std::endl;
        auto inputLayer = input.second->getInputData()->getCreatorLayer().lock();

        if (dataPtrs.find(input.second->getInputData().get()) == dataPtrs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (dataPtrs.find(input.second->getInputData().get()) == dataPtrs.end()) {" << std::endl;
            dataPtrs.insert(input.second->getInputData().get());
        }

        if (!inputLayer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (!inputLayer) {" << std::endl;
            // For v1 parser
            inputLayer.reset(new CNNLayer(
                {input.second->getInputData()->getName(), "Input", input.second->getInputData()->getPrecision()}));

            inputLayer->outData.push_back(input.second->getInputData());
        }
        const auto layer =
            InputLayer(inputLayer->name).setPort(Port(inputLayer->outData[0]->getTensorDesc().getDims()));
        name2id[layer.getName()] = addLayer(layer);

        for (const auto& nlayer : input.second->getInputData()->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& nlayer : input.second->getInputData()->getInputTo()) {" << std::endl;
            queueLayers.push_back(nlayer.second);
        }
    }
    for (auto input : allInputs) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (auto input : allInputs) {" << std::endl;
        auto isRealInput =
            std::find_if(std::begin(inputs), std::end(inputs), [&](InputsDataMap::value_type& inputInfo) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              std::find_if(std::begin(inputs), std::end(inputs), [&](InputsDataMap::value_type& inputInfo) {" << std::endl;
                return inputInfo.second->getInputData()->getName() == input->name;
            });
        if (isRealInput != std::end(inputs)) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (isRealInput != std::end(inputs)) {" << std::endl;
            continue;
        }

        details::CaselessEq<std::string> eq;
        CNNLayerPtr cnnLayer = input;

        if (eq(input->type, "Memory")) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (eq(input->type, 'Memory')) {" << std::endl;
            auto memoryId = input->GetParamAsString("id");
            cnnLayer.reset(new CNNLayer({input->name + "/id=" + memoryId, "MemoryInput", input->precision}));
            cnnLayer->params = input->params;
            cnnLayer->outData = input->outData;
        }

        createGenericFromCNNLayer(cnnLayer);

        size_t count_out = 0;
        for (auto&& outData : input->outData) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (auto&& outData : input->outData) {" << std::endl;
            for (auto&& nlayer : outData->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (auto&& nlayer : outData->getInputTo()) {" << std::endl;
                queueLayers.push_back(nlayer.second);
            }
            count_out++;
        }
    }
    while (!queueLayers.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      while (!queueLayers.empty()) {" << std::endl;
        auto cnnLayerPtr = *queueLayers.begin();

        if (name2id.find(cnnLayerPtr->name) == name2id.end()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (name2id.find(cnnLayerPtr->name) == name2id.end()) {" << std::endl;
            createGenericFromCNNLayer(cnnLayerPtr);

            for (auto&& outData : cnnLayerPtr->outData) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (auto&& outData : cnnLayerPtr->outData) {" << std::endl;
                for (auto&& nlayer : outData->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                  for (auto&& nlayer : outData->getInputTo()) {" << std::endl;
                    queueLayers.push_back(nlayer.second);
                }
            }
        }

        queueLayers.erase(queueLayers.begin());
    }
    std::map<std::string, DataPtr> output;
    network.getOutputsInfo(output);

    for (auto it = output.begin(); it != output.end(); it++) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (auto it = output.begin(); it != output.end(); it++) {" << std::endl;
        CNNLayerPtr creator = (*it).second->getCreatorLayer().lock();
        if (name2id.find(creator->name) == name2id.end())
            THROW_IE_EXCEPTION << "Cannot find output layer " << creator->name;

        auto lastLayer = getLayer(name2id[creator->name]);
        if (lastLayer->getName() == "" && lastLayer->getType().empty())
            THROW_IE_EXCEPTION << "Cannot find output layer " << creator->name;

        std::string name = "out_" + lastLayer->getName();

        CNNLayerPtr cnnOutLayer(new CNNLayer({name, "Output", creator->outData[0]->getPrecision()}));
        cnnOutLayer->insData.push_back((*it).second);

        idx_t outLayerId = createGenericFromCNNLayer(cnnOutLayer);

        idx_t inIdx(0);
        for (size_t i = 0; i < creator->outData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (size_t i = 0; i < creator->outData.size(); i++) {" << std::endl;
            if (creator->outData[i] == (*it).second) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (creator->outData[i] == (*it).second) {" << std::endl;
                inIdx = i;
                break;
            }
        }

        parameters["connections"].as<std::vector<Connection>>().push_back(
            Connection({lastLayer->getId(), inIdx}, {outLayerId}));
    }

    for (const auto dataPtr : dataPtrs) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (const auto dataPtr : dataPtrs) {" << std::endl;
        auto cnnInputLayer = dataPtr->getCreatorLayer().lock();
        idx_t inIdx(0);
        if (!cnnInputLayer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (!cnnInputLayer) {" << std::endl;
            // For v1 parser
            cnnInputLayer.reset(new CNNLayer({dataPtr->getName(), "Input", dataPtr->getPrecision()}));
        } else {
            for (size_t i = 0; i < cnnInputLayer->outData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (size_t i = 0; i < cnnInputLayer->outData.size(); i++) {" << std::endl;
                if (cnnInputLayer->outData[i].get() == dataPtr) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                  if (cnnInputLayer->outData[i].get() == dataPtr) {" << std::endl;
                    inIdx = i;
                    break;
                }
            }
        }
        for (const auto& it : dataPtr->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& it : dataPtr->getInputTo()) {" << std::endl;
            if (name2id.find(cnnInputLayer->name) == name2id.end() || name2id.find(it.second->name) == name2id.end())
                THROW_IE_EXCEPTION << "Cannot create connections between nodes: " << cnnInputLayer->name << " -> "
                                   << it.second->name;
            idx_t outIdx(0);

            for (size_t i = 0; i < it.second->insData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (size_t i = 0; i < it.second->insData.size(); i++) {" << std::endl;
                const auto lockedData = it.second->insData[i].lock();
                if (lockedData && lockedData.get() == dataPtr) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                  if (lockedData && lockedData.get() == dataPtr) {" << std::endl;
                    outIdx = i;
                    break;
                }
            }
            parameters["connections"].as<std::vector<Connection>>().push_back(
                Connection({name2id[cnnInputLayer->name], inIdx}, {name2id[it.second->name], outIdx}));
        }
    }

    for (const auto& input : inputs) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (const auto& input : inputs) {" << std::endl;
        addPreProcessFor(input.second);
    }
}

const std::vector<Builder::Layer::Ptr>& Builder::Network::getLayers() const {
    return parameters.at("layers").as<std::vector<Layer::Ptr>>();
}
std::vector<Builder::Layer::Ptr>& Builder::Network::getLayers() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  std::vector<Builder::Layer::Ptr>& Builder::Network::getLayers() {" << std::endl;
    return parameters["layers"].as<std::vector<Layer::Ptr>>();
}

idx_t Builder::Network::addLayer(const std::vector<PortInfo>& inputs, const Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  idx_t Builder::Network::addLayer(const std::vector<PortInfo>& inputs, const Layer& layer) {" << std::endl;
    IE_PROFILING_AUTO_SCOPE(Builder::Network::addLayer)
    auto layer_id = addLayer(layer);
    for (size_t i = 0; i < inputs.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (size_t i = 0; i < inputs.size(); i++) {" << std::endl;
        connect({inputs[i].layerId(), inputs[i].portId()}, {layer_id, i});
    }
    return layer_id;
}

idx_t Builder::Network::addLayer(const Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  idx_t Builder::Network::addLayer(const Layer& layer) {" << std::endl;
    auto getAvailableId = [&](idx_t defaultId) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      auto getAvailableId = [&](idx_t defaultId) {" << std::endl;
        if (defaultId == (std::numeric_limits<idx_t>::max)()) defaultId = 0;

        auto it = parameters["layers"].as<std::vector<Layer::Ptr>>().begin();
        while (it != parameters["layers"].as<std::vector<Layer::Ptr>>().end()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          while (it != parameters['layers'].as<std::vector<Layer::Ptr>>().end()) {" << std::endl;
            for (it = parameters["layers"].as<std::vector<Layer::Ptr>>().begin();
                 it != parameters["layers"].as<std::vector<Layer::Ptr>>().end(); it++) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                   it != parameters['layers'].as<std::vector<Layer::Ptr>>().end(); it++) {" << std::endl;
                if ((*it)->getId() == defaultId) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                  if ((*it)->getId() == defaultId) {" << std::endl;
                    defaultId++;
                    break;
                }
            }
        }
        return defaultId;
    };
    auto generateAvailableName = [&](const std::string& name, idx_t id) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      auto generateAvailableName = [&](const std::string& name, idx_t id) {" << std::endl;
        const std::string idName = "id" + std::to_string(id);
        std::string generatedName(name);
        if (generatedName.empty()) generatedName = idName;
        bool nameIsUnique(false);
        while (!nameIsUnique) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          while (!nameIsUnique) {" << std::endl;
            nameIsUnique = true;
            for (const auto& layer : parameters["layers"].as<std::vector<Layer::Ptr>>()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (const auto& layer : parameters['layers'].as<std::vector<Layer::Ptr>>()) {" << std::endl;
                if (generatedName == layer->getName()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                  if (generatedName == layer->getName()) {" << std::endl;
                    nameIsUnique = false;
                    generatedName += "_" + idName;
                }
            }
        }
        return generatedName;
    };
    idx_t generatedId = getAvailableId(layer.getId());
    const auto name = generateAvailableName(layer.getName(), generatedId);
    parameters["layers"].as<std::vector<Layer::Ptr>>().emplace_back(std::make_shared<Layer>(generatedId, layer));
    parameters["layers"]
        .as<std::vector<Layer::Ptr>>()[parameters["layers"].as<std::vector<Layer::Ptr>>().size() - 1]
        ->setName(name);
    return generatedId;
}

void Builder::Network::connect(const PortInfo& input, const PortInfo& output) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  void Builder::Network::connect(const PortInfo& input, const PortInfo& output) {" << std::endl;
    const auto mergePortData = [&]() -> bool {
        const auto blobEqualOrEmpty = [](const Blob::Ptr& ref, const Blob::Ptr& test) -> bool {
            return (ref->size() == test->size() || test->size() == 0) &&
                   (!memcmp(ref->cbuffer(), test->cbuffer(), test->byteSize())) &&
                   (ref->getTensorDesc().getPrecision() == test->getTensorDesc().getPrecision() ||
                    test->getTensorDesc().getPrecision() == Precision::UNSPECIFIED) &&
                   (ref->getTensorDesc().getLayout() == test->getTensorDesc().getLayout() ||
                    test->getTensorDesc().getLayout() == Layout::ANY) &&
                   (ref->getTensorDesc().getDims() == test->getTensorDesc().getDims() ||
                    test->getTensorDesc().getDims().empty()) &&
                   (ref->cbuffer().as<char*>() == test->cbuffer().as<char*>() || test->cbuffer() == nullptr);
        };

        const auto srcPortData = getLayer(input.layerId())->getOutputPorts()[input.portId()].getData();
        const auto dstPortData = getLayer(output.layerId())->getInputPorts()[output.portId()].getData();
        if (srcPortData == dstPortData) return true;

        if (srcPortData->getParameters() != dstPortData->getParameters() && !srcPortData->getParameters().empty() &&
            !dstPortData->getParameters().empty())
            return false;

        size_t srcDataCount(0), dstDataCount(0);
        if (!srcPortData->getParameters().empty()) srcDataCount++;
        if (!dstPortData->getParameters().empty()) dstDataCount++;

        const auto srcBlb = srcPortData->getData();
        const auto dstBlb = dstPortData->getData();
        if (srcBlb == dstBlb ||
            (srcBlb->size() == dstBlb->size() && srcBlb->getTensorDesc() == dstBlb->getTensorDesc() &&
             ((srcBlb->cbuffer().as<char*>() == dstBlb->cbuffer().as<char*>()) ||
              (srcBlb->cbuffer() != nullptr && dstBlb->cbuffer() != nullptr &&
               !memcmp(srcBlb->cbuffer(), dstBlb->cbuffer(), dstBlb->byteSize()))))) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                 !memcmp(srcBlb->cbuffer(), dstBlb->cbuffer(), dstBlb->byteSize()))))) {" << std::endl;
            srcDataCount++;
            dstDataCount++;
        } else if (blobEqualOrEmpty(srcBlb, dstBlb)) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          } else if (blobEqualOrEmpty(srcBlb, dstBlb)) {" << std::endl;
            srcDataCount++;
        } else if (blobEqualOrEmpty(dstBlb, srcBlb)) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          } else if (blobEqualOrEmpty(dstBlb, srcBlb)) {" << std::endl;
            dstDataCount++;
        } else {
            return false;
        }

        if (dstDataCount > srcDataCount) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (dstDataCount > srcDataCount) {" << std::endl;
            // Change source and all src destination data
            for (const auto& connection : getLayerConnections(input.layerId())) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (const auto& connection : getLayerConnections(input.layerId())) {" << std::endl;
                if (connection.from() != input) continue;
                getLayer(connection.to().layerId())->getInputPorts()[connection.to().portId()].setData(dstPortData);
            }
            getLayer(input.layerId())->getOutputPorts()[input.portId()].setData(dstPortData);
        } else {
            // Change destination data
            getLayer(output.layerId())->getInputPorts()[output.portId()].setData(srcPortData);
        }

        return true;
    };

    if (!mergePortData()) THROW_IE_EXCEPTION << "Cannot connect two ports with different data!";

    parameters["connections"].as<std::vector<Connection>>().emplace_back(input, output);
}

void Builder::Network::removeLayer(idx_t layerId) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  void Builder::Network::removeLayer(idx_t layerId) {" << std::endl;
    auto it = parameters["layers"].as<std::vector<Layer::Ptr>>().begin();
    for (; it != parameters["layers"].as<std::vector<Layer::Ptr>>().end(); it++) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (; it != parameters['layers'].as<std::vector<Layer::Ptr>>().end(); it++) {" << std::endl;
        if ((*it)->getId() == layerId) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if ((*it)->getId() == layerId) {" << std::endl;
            break;
        }
    }
    if (it != parameters["layers"].as<std::vector<Layer::Ptr>>().end())
        parameters["layers"].as<std::vector<Layer::Ptr>>().erase(it);
}

void Builder::Network::disconnect(const Connection& connection) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  void Builder::Network::disconnect(const Connection& connection) {" << std::endl;
    auto it = parameters["connections"].as<std::vector<Connection>>().begin();
    for (; it != parameters["connections"].as<std::vector<Connection>>().end(); it++) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (; it != parameters['connections'].as<std::vector<Connection>>().end(); it++) {" << std::endl;
        if (connection == *it) break;
    }
    if (it != parameters["connections"].as<std::vector<Connection>>().end())
        parameters["connections"].as<std::vector<Connection>>().erase(it);

    try {
        auto layer = getLayer(connection.to().layerId());
        layer->getInputPorts()[connection.to().portId()].setData(std::make_shared<PortData>());
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      } catch (InferenceEngine::details::InferenceEngineException& ex) {" << std::endl;
    }
}

const INetwork::CPtr Builder::Network::build() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  const INetwork::CPtr Builder::Network::build() {" << std::endl;
    validate();
    InferenceEngine::Builder::Network::Ptr network =
        std::make_shared<InferenceEngine::Builder::Network>(static_cast<const INetwork&>(*this));
    return network;
}

void Builder::Network::validate() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  void Builder::Network::validate() {" << std::endl;
    // Check that all ports are connected
    for (const auto& layer : getLayers()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (const auto& layer : getLayers()) {" << std::endl;
        std::vector<bool> existInCon(layer->getInputPorts().size());
        for (size_t i = 0; i < layer->getInputPorts().size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (size_t i = 0; i < layer->getInputPorts().size(); i++) {" << std::endl;
            if (layer->getInputPorts()[i].getParameters().find("type") !=
                layer->getInputPorts()[i].getParameters().end())
                existInCon[i] = true;
        }
        std::vector<bool> existOutCon(layer->getOutputPorts().size());

        const auto layerConnections = getLayerConnections(layer->getId());
        for (const auto& connection : layerConnections) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& connection : layerConnections) {" << std::endl;
            if (connection.from().layerId() == layer->getId()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (connection.from().layerId() == layer->getId()) {" << std::endl;
                existOutCon[connection.from().portId()] = true;
                getLayer(connection.to().layerId());
            }
            if (connection.to().layerId() == layer->getId()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (connection.to().layerId() == layer->getId()) {" << std::endl;
                existInCon[connection.to().portId()] = true;
                getLayer(connection.from().layerId());
            }
        }
        bool allPortsConnected = true;
        for (const auto& cons : {existInCon, existOutCon}) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& cons : {existInCon, existOutCon}) {" << std::endl;
            for (const auto& existCon : cons) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (const auto& existCon : cons) {" << std::endl;
                allPortsConnected = allPortsConnected && existCon;
            }
        }
        if (!allPortsConnected)
            THROW_IE_EXCEPTION << "Not all ports of layer " << layer->getName() << " were connected!";
    }

    // Check all layers
    for (const auto& connection : getConnections()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (const auto& connection : getConnections()) {" << std::endl;
        if (!getLayer(connection.to().layerId()))
            THROW_IE_EXCEPTION << "Cannot find layer with id: " << connection.to().layerId();
        if (!getLayer(connection.from().layerId()))
            THROW_IE_EXCEPTION << "Cannot find layer with id: " << connection.from().layerId();
    }

    std::map<std::string, SizeVector> inputShapes;
    for (const auto& input : getInputs()) inputShapes[input->getName()] = input->getOutputPorts()[0].shape();

    ShapeInfer::Reshaper reshaper(this);
    ResponseDesc resp;
    StatusCode sts = reshaper.run(inputShapes, &resp);
    // Not all implementations may be registered if all shapes were read from IR.
    if (sts == NOT_FOUND) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      if (sts == NOT_FOUND) {" << std::endl;
        bool allShapesLooksGood = true;
        for (const auto& connection : getConnections()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& connection : getConnections()) {" << std::endl;
            if (getLayer(connection.from().layerId())->getOutputPorts()[connection.from().portId()].shape() !=
                    getLayer(connection.to().layerId())->getInputPorts()[connection.to().portId()].shape() ||
                getLayer(connection.to().layerId())->getInputPorts()[connection.to().portId()].shape().empty()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                  getLayer(connection.to().layerId())->getInputPorts()[connection.to().portId()].shape().empty()) {" << std::endl;
                allShapesLooksGood = false;
                break;
            }
        }
        if (allShapesLooksGood) sts = OK;
    }

    if (sts != OK) THROW_IE_EXCEPTION << resp.msg;

    // Check all parameters
    for (const auto& layer : getLayers()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (const auto& layer : getLayers()) {" << std::endl;
        try {
            layer->build();
        } catch (InferenceEngine::details::InferenceEngineException& ex) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          } catch (InferenceEngine::details::InferenceEngineException& ex) {" << std::endl;
            THROW_IE_EXCEPTION << "Cannot build layer " << layer->getName() << ": " << ex.what();
        } catch (std::bad_cast& ex) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          } catch (std::bad_cast& ex) {" << std::endl;
            THROW_IE_EXCEPTION << "Cannot build layer " << layer->getName() << ": " << ex.what();
        }
    }
}

Builder::Network::operator const INetwork::CPtr() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Network::operator const INetwork::CPtr() {" << std::endl;
    return build();
}

const ILayer::CPtr Builder::Network::getLayer(idx_t layerId) const noexcept {
    try {
        for (auto& layer : getLayers()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (auto& layer : getLayers()) {" << std::endl;
            if (layer->getId() == layerId) return layer->build();
        }
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      } catch (...) {" << std::endl;
    }

    return nullptr;
}

Builder::Layer::Ptr Builder::Network::getLayer(idx_t layerId) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Layer::Ptr Builder::Network::getLayer(idx_t layerId) {" << std::endl;
    for (auto& layer : getLayers()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (auto& layer : getLayers()) {" << std::endl;
        if (layer->getId() == layerId) return layer;
    }
    THROW_IE_EXCEPTION << "Cannot find layer with id: " << layerId;
}

const std::string& Builder::Network::getName() const noexcept {
    static std::string errName;
    try {
        return parameters.at("name");
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      } catch (...) {" << std::endl;
        return errName;
    }
}

const Context& Builder::Network::getContext() const noexcept {
    static Context errCtx;
    try {
        return parameters.at("context");
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      } catch (...) {" << std::endl;
        return errCtx;
    }
}

Context& Builder::Network::getContext() noexcept {
    static Context errCtx;
    try {
        return parameters.at("context");
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      } catch (...) {" << std::endl;
        return errCtx;
    }
}

Builder::Network::const_iterator Builder::Network::begin() const noexcept {
    try {
        return Network::const_iterator(this);
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      } catch (...) {" << std::endl;
        return Network::const_iterator(this, true);
    }
}

Builder::Network::const_iterator Builder::Network::end() const noexcept {
    return Network::const_iterator(this, true);
}

size_t Builder::Network::size() const noexcept {
    return static_cast<size_t>(std::distance(std::begin(*this), std::end(*this)));
}

Builder::Network::iterator Builder::Network::begin() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Network::iterator Builder::Network::begin() {" << std::endl;
    return Network::iterator(this);
}

Builder::Network::iterator Builder::Network::end() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  Builder::Network::iterator Builder::Network::end() {" << std::endl;
    return Network::iterator(this, true);
}

const std::vector<ILayer::CPtr> Builder::Network::getInputs() const noexcept {
    std::vector<ILayer::CPtr> inputs;
    try {
        for (const auto& layer : parameters.at("layers").as<std::vector<Layer::Ptr>>()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& layer : parameters.at('layers').as<std::vector<Layer::Ptr>>()) {" << std::endl;
            bool isInputLayer = true;
            for (const auto& connection : getLayerConnections(layer->getId())) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (const auto& connection : getLayerConnections(layer->getId())) {" << std::endl;
                if (connection.to().layerId() == layer->getId()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                  if (connection.to().layerId() == layer->getId()) {" << std::endl;
                    isInputLayer = false;
                    break;
                }
            }
            if (isInputLayer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (isInputLayer) {" << std::endl;
                inputs.push_back(layer->build());
            }
        }
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      } catch (...) {" << std::endl;
    }
    return inputs;
}

std::vector<Builder::Layer::Ptr> Builder::Network::getInputs() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  std::vector<Builder::Layer::Ptr> Builder::Network::getInputs() {" << std::endl;
    std::vector<Builder::Layer::Ptr> inputs;
    for (auto& layer : parameters.at("layers").as<std::vector<Layer::Ptr>>()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (auto& layer : parameters.at('layers').as<std::vector<Layer::Ptr>>()) {" << std::endl;
        bool isInputLayer = true;
        for (const auto& connection : getLayerConnections(layer->getId())) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& connection : getLayerConnections(layer->getId())) {" << std::endl;
            if (connection.to().layerId() == layer->getId()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (connection.to().layerId() == layer->getId()) {" << std::endl;
                isInputLayer = false;
                break;
            }
        }
        if (isInputLayer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (isInputLayer) {" << std::endl;
            inputs.push_back(layer);
        }
    }
    return inputs;
}

const std::vector<ILayer::CPtr> Builder::Network::getOutputs() const noexcept {
    std::vector<ILayer::CPtr> outputs;
    try {
        for (const auto& layer : parameters.at("layers").as<std::vector<Layer::Ptr>>()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& layer : parameters.at('layers').as<std::vector<Layer::Ptr>>()) {" << std::endl;
            bool isOutputLayer = true;
            for (const auto& connection : getLayerConnections(layer->getId())) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              for (const auto& connection : getLayerConnections(layer->getId())) {" << std::endl;
                if (connection.from().layerId() == layer->getId()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:                  if (connection.from().layerId() == layer->getId()) {" << std::endl;
                    isOutputLayer = false;
                    break;
                }
            }
            if (isOutputLayer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (isOutputLayer) {" << std::endl;
                outputs.push_back(layer->build());
            }
        }
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      } catch (...) {" << std::endl;
    }
    return outputs;
}

std::vector<Builder::Layer::Ptr> Builder::Network::getOutputs() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:  std::vector<Builder::Layer::Ptr> Builder::Network::getOutputs() {" << std::endl;
    std::vector<Builder::Layer::Ptr> outputs;
    for (auto& layer : parameters.at("layers").as<std::vector<Layer::Ptr>>()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      for (auto& layer : parameters.at('layers').as<std::vector<Layer::Ptr>>()) {" << std::endl;
        bool isOutputLayer = true;
        for (const auto& connection : getLayerConnections(layer->getId())) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto& connection : getLayerConnections(layer->getId())) {" << std::endl;
            if (connection.from().layerId() == layer->getId()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:              if (connection.from().layerId() == layer->getId()) {" << std::endl;
                isOutputLayer = false;
                break;
            }
        }
        if (isOutputLayer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          if (isOutputLayer) {" << std::endl;
            outputs.push_back(layer);
        }
    }
    return outputs;
}

const std::vector<Connection>& Builder::Network::getConnections() const {
    return parameters.at("connections").as<std::vector<Connection>>();
}

const std::vector<Connection> Builder::Network::getLayerConnections(idx_t layerId) const noexcept {
    std::vector<Connection> layerConnections;
    try {
        for (const auto connection : parameters.at("connections").as<std::vector<Connection>>()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:          for (const auto connection : parameters.at('connections').as<std::vector<Connection>>()) {" << std::endl;
            if (connection.from().layerId() == layerId || connection.to().layerId() == layerId)
                layerConnections.push_back(connection);
        }
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_network_builder.cpp:      } catch (...) {" << std::endl;
    }
    return layerConnections;
}
