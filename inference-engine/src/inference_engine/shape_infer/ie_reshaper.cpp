#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_infer/ie_reshaper.hpp"

#include <debug.h>
#include <ie_layers.h>

#include <blob_factory.hpp>
#include <builders/ie_split_layer.hpp>
#include <functional>
#include <graph_tools.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "details/caseless.hpp"
#include "details/ie_cnn_network_tools.h"
#include "ie_cnn_layer_builder.h"
#include "ie_reshaper.hpp"
#include "shape_infer/built-in/ie_built_in_holder.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ShapeInfer;

IE_SUPPRESS_DEPRECATED_START

Reshaper::Reshaper(Builder::Network* network): network(network) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  Reshaper::Reshaper(Builder::Network* network): network(network) {" << std::endl;}

IE_SUPPRESS_DEPRECATED_END

inline static std::vector<CNNLayerPtr> SortTopologicallyStartsFrom(const std::vector<DataPtr>& inputs) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  inline static std::vector<CNNLayerPtr> SortTopologicallyStartsFrom(const std::vector<DataPtr>& inputs) {" << std::endl;
    std::vector<CNNLayerPtr> all_layers;
    CNNNetForestDFS(
        inputs,
        [&](CNNLayerPtr current) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          [&](CNNLayerPtr current) {" << std::endl;
            all_layers.push_back(current);
        },
        false);
    std::reverse(all_layers.begin(), all_layers.end());
    return all_layers;
}

Reshaper::Reshaper(std::vector<DataPtr> insDatas, const LauncherCreator::Ptr& launcherCreator): network(nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  Reshaper::Reshaper(std::vector<DataPtr> insDatas, const LauncherCreator::Ptr& launcherCreator): network(nullptr) {" << std::endl;
    auto builtIn = std::make_shared<BuiltInShapeInferHolder>();
    _allTypes = getTypeNamesFromExtension(builtIn);
    _extensions.push_back(builtIn);

    _allSortedLayers = SortTopologicallyStartsFrom(insDatas);
    for (auto& in_data : insDatas) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto& in_data : insDatas) {" << std::endl;
        for (auto layer : in_data->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (auto layer : in_data->getInputTo()) {" << std::endl;
            _inputLayers.insert(layer.second);
        }
    }

    if (_inputLayers.empty() || _allSortedLayers.empty())
        THROW_IE_EXCEPTION << "Unsupported model for shape inference: failed to collect inputs and layers";

    for (auto const& currentLayer : _allSortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto const& currentLayer : _allSortedLayers) {" << std::endl;
        auto createdLauncher = launcherCreator->createNotInputLauncher(currentLayer.get(), _extensions);
        _launchers.insert(createdLauncher);
    }
}

Reshaper::Reshaper(ICNNNetwork& network, const LauncherCreator::Ptr& launcherCreator): network(nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  Reshaper::Reshaper(ICNNNetwork& network, const LauncherCreator::Ptr& launcherCreator): network(nullptr) {" << std::endl;
    auto builtIn = std::make_shared<BuiltInShapeInferHolder>();
    _allTypes = getTypeNamesFromExtension(builtIn);
    _extensions.push_back(builtIn);

    auto inputLayers = CNNNetGetAllInputLayers(network);
    for (const auto& layer : inputLayers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (const auto& layer : inputLayers) {" << std::endl;
        _inputLayers.insert(layer);
    }

    _allSortedLayers = CNNNetSortTopologically(network);
    if (_inputLayers.empty() || _allSortedLayers.empty())
        THROW_IE_EXCEPTION << "Unsupported model for shape inference: failed to collect inputs and layers";
    for (auto const& currentLayer : _allSortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto const& currentLayer : _allSortedLayers) {" << std::endl;
        auto foundInput =
            std::find_if(_inputLayers.begin(), _inputLayers.end(), [&currentLayer](const CNNLayerPtr& inputLayer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              std::find_if(_inputLayers.begin(), _inputLayers.end(), [&currentLayer](const CNNLayerPtr& inputLayer) {" << std::endl;
                return currentLayer->name == inputLayer->name;
            });
        ReshapeLauncher::Ptr createdLauncher;
        if (foundInput == _inputLayers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          if (foundInput == _inputLayers.end()) {" << std::endl;
            createdLauncher = launcherCreator->createNotInputLauncher(currentLayer.get(), _extensions);
        } else {
            createdLauncher = launcherCreator->createInputLauncher(currentLayer.get(), _extensions);
        }
        _launchers.insert(createdLauncher);
    }
}

void Reshaper::AddExtension(const IShapeInferExtensionPtr& extension) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  void Reshaper::AddExtension(const IShapeInferExtensionPtr& extension) {" << std::endl;
    if (!extension) THROW_IE_EXCEPTION << "Failed to add empty shape infer extension";

    if (network) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      if (network) {" << std::endl;
        IE_SUPPRESS_DEPRECATED_START
        network->getContext().addExtension(extension);
        IE_SUPPRESS_DEPRECATED_END
        return;
    }

    auto newLayerTypes = getTypeNamesFromExtension(extension);
    std::string badLayerTypes;
    for (const auto& type : newLayerTypes) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (const auto& type : newLayerTypes) {" << std::endl;
        auto ret = _allTypes.insert(type);
        if (!ret.second) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          if (!ret.second) {" << std::endl;
            if (!badLayerTypes.empty()) badLayerTypes += ", ";
            badLayerTypes += type;
        }
    }
    if (!badLayerTypes.empty())
        THROW_IE_EXCEPTION << "Failed to add extension with already registered types:" << badLayerTypes;

    for (auto const& layerType : newLayerTypes) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto const& layerType : newLayerTypes) {" << std::endl;
        auto foundLauncher = _launchers.begin();
        // find all layers with given type
        std::vector<ReshapeLauncher::Ptr> launchersToInsert;
        while (foundLauncher != _launchers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          while (foundLauncher != _launchers.end()) {" << std::endl;
            foundLauncher =
                std::find_if(foundLauncher, _launchers.end(), [&layerType](const ReshapeLauncher::Ptr& launcher) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:                  std::find_if(foundLauncher, _launchers.end(), [&layerType](const ReshapeLauncher::Ptr& launcher) {" << std::endl;
                    return layerType == launcher->getLayerType();
                });
            if (foundLauncher != _launchers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              if (foundLauncher != _launchers.end()) {" << std::endl;
                IShapeInferImpl::Ptr impl;
                StatusCode sts = extension->getShapeInferImpl(impl, layerType.c_str(), nullptr);
                if (sts == OK && impl != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:                  if (sts == OK && impl != nullptr) {" << std::endl;
                    auto newLauncher = std::make_shared<ReshapeLauncher>((*foundLauncher)->getLayer(), impl);
                    newLauncher->setShapeInferImpl(impl);
                    launchersToInsert.push_back(newLauncher);
                    foundLauncher = _launchers.erase(foundLauncher);
                } else {
                    THROW_IE_EXCEPTION << "Failed to get registered Shape Infer Implementation for type: " << layerType;
                }
            }
        }
        for (const auto& launcher : launchersToInsert) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (const auto& launcher : launchersToInsert) {" << std::endl;
            _launchers.insert(launcher);
        }
    }
    _extensions.push_back(extension);
}

ReshapeLauncher::Ptr Reshaper::getLauncherByLayerName(const std::string& layerName) const {
    auto foundLauncher =
        std::find_if(_launchers.begin(), _launchers.end(), [&layerName](const ReshapeLauncher::Ptr& launcher) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          std::find_if(_launchers.begin(), _launchers.end(), [&layerName](const ReshapeLauncher::Ptr& launcher) {" << std::endl;
            return launcher->getLayerName() == layerName;
        });
    if (foundLauncher == _launchers.end())
        THROW_IE_EXCEPTION << "Failed to reshape layer ('" << layerName << "'): can't find the corresponding launcher";
    return *foundLauncher;
}

StatusCode Reshaper::run(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  StatusCode Reshaper::run(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp) {" << std::endl;
    if (network) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      if (network) {" << std::endl;
        return networkShapeInfer(inputShapes, resp);
    }

    // WA: In another case we should change the registration logic of shape implementations
    static std::mutex reshapeMutex;
    {
        std::lock_guard<std::mutex> lock(reshapeMutex);
        // Reset all shapes from previous run
        for (const auto& launcher : _launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (const auto& launcher : _launchers) {" << std::endl;
            launcher->reset();
        }

        // Set new input shapes
        for (auto const& input : _inputLayers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (auto const& input : _inputLayers) {" << std::endl;
            std::string layerName = input->name;
            for (auto const& outData : input->outData) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              for (auto const& outData : input->outData) {" << std::endl;
                std::string dataName = outData->getName();
                auto foundShapeIt = inputShapes.find(dataName);
                auto foundLauncher = getLauncherByLayerName(layerName);
                if (foundShapeIt != inputShapes.end()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:                  if (foundShapeIt != inputShapes.end()) {" << std::endl;
                    foundLauncher->setShapeByName(foundShapeIt->second, dataName);
                } else {
                    foundLauncher->setIRShapeByName(dataName);
                }
            }
        }

        // do reshape
        for (auto& layer : _allSortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (auto& layer : _allSortedLayers) {" << std::endl;
            auto foundLauncher = getLauncherByLayerName(layer->name);
            foundLauncher->reshape(_launchers);
            foundLauncher->constInfer(_launchers);
        }

        // apply changes
        for (auto& layer : _allSortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (auto& layer : _allSortedLayers) {" << std::endl;
            auto foundLauncher = getLauncherByLayerName(layer->name);
            foundLauncher->applyChanges(layer.get());
        }
        return OK;
    }
}

StatusCode Reshaper::runNoApply(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  StatusCode Reshaper::runNoApply(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp) {" << std::endl;
    // Reset all shapes from previous run
    for (const auto& launcher : _launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (const auto& launcher : _launchers) {" << std::endl;
        launcher->reset();
    }

    // Set new input shapes
    for (auto const& input : _inputLayers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto const& input : _inputLayers) {" << std::endl;
        std::string layerName = input->name;
        for (auto const& inData_w : input->insData) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (auto const& inData_w : input->insData) {" << std::endl;
            auto inData = inData_w.lock();
            auto dataName = inData->getName();
            auto foundShapeIt = inputShapes.find(dataName);
            auto foundLauncher = getLauncherByLayerName(layerName);
            if (foundShapeIt != inputShapes.end()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              if (foundShapeIt != inputShapes.end()) {" << std::endl;
                foundLauncher->setShapeByName(foundShapeIt->second, dataName);
            } else {
                foundLauncher->setIRShapeByName(dataName);
            }
        }
    }

    // do reshape
    for (auto& layer : _allSortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto& layer : _allSortedLayers) {" << std::endl;
        auto foundLauncher = getLauncherByLayerName(layer->name);
        foundLauncher->reshape(_launchers);
    }
    return OK;
}

StatusCode Reshaper::apply(ResponseDesc* resp) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  StatusCode Reshaper::apply(ResponseDesc* resp) {" << std::endl;
    // apply changes
    for (auto& layer : _allSortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto& layer : _allSortedLayers) {" << std::endl;
        auto foundLauncher = getLauncherByLayerName(layer->name);
        foundLauncher->applyChanges(layer.get());
    }
    return OK;
}

SizeVector Reshaper::getResultShapeFor(DataPtr& data, ResponseDesc* resp) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  SizeVector Reshaper::getResultShapeFor(DataPtr& data, ResponseDesc* resp) {" << std::endl;
    auto creator_layer = data->getCreatorLayer().lock();
    std::string creator_layer_name;
    if (creator_layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      if (creator_layer) {" << std::endl;
        creator_layer_name = creator_layer->name;
    }
    auto foundLauncher = getLauncherByLayerName(creator_layer_name);
    return foundLauncher->getShapeByName(data->getName());
}

StatusCode Reshaper::networkShapeInfer(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  StatusCode Reshaper::networkShapeInfer(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp) {" << std::endl;
    if (!network) return DescriptionBuffer(GENERAL_ERROR, resp) << "Cannot infer shapes! Network is not loaded.";

    IE_SUPPRESS_DEPRECATED_START

    std::vector<Builder::Layer> propagatedLayers;
    Builder::Network propagatedNetwork(*network);

    // Set new input shapes
    for (auto& layer : propagatedNetwork) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto& layer : propagatedNetwork) {" << std::endl;
        if (inputShapes.find(layer->getName()) == inputShapes.end() ||
            details::CaselessEq<std::string>()(layer->getType(), "Const"))
            continue;

        if (layer->getOutputPorts().size() != 1)
            return DescriptionBuffer(GENERAL_ERROR, resp)
                   << "Cannot infer shapes! Input layers can have only one output port.";

        layer->getOutputPorts()[0].setShape(inputShapes.find(layer->getName())->second);
    }

    std::map<idx_t, std::map<std::string, std::string>> preparedParams;
    // Prepare params for split layer
    for (auto& layer : propagatedNetwork) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto& layer : propagatedNetwork) {" << std::endl;
        if ((layer->getType() == "Reshape" || layer->getType() == "Flatten") && layer->getInputPorts().size() != 2 &&
            !layer->getInputPorts()[0].shape().empty() &&
            layer->getParameters().find("axis") != layer->getParameters().end() &&
            (layer->getParameters().find("dim") == layer->getParameters().end() ||
             layer->getParameters().at("dim").as<std::vector<int>>().empty())) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:               layer->getParameters().at('dim').as<std::vector<int>>().empty())) {" << std::endl;
            auto inputShape = layer->getInputPorts()[0].shape();
            size_t inputShapeTotal =
                std::accumulate(inputShape.begin(), inputShape.end(), 1lu, std::multiplies<size_t>());
            std::vector<int> dim;
            size_t axis = layer->getParameters().at("axis");
            for (size_t i = 0; i < axis; i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              for (size_t i = 0; i < axis; i++) {" << std::endl;
                dim.emplace_back(inputShape[i]);
                inputShapeTotal /= inputShape[i];
            }
            if (dim.size() < inputShape.size()) dim.emplace_back(inputShapeTotal);
            layer->getParameters()["dim"] = dim;
        }

        std::map<std::string, std::string> params =
            InferenceEngine::Builder::convertParameters2Strings(layer->getParameters());
        if (layer->getType() == "Split") {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          if (layer->getType() == 'Split') {" << std::endl;
            Builder::SplitLayer splitLayer(layer);
            std::vector<size_t> sizes;
            size_t axisSize = splitLayer.getInputPort().shape()[splitLayer.getAxis()];
            size_t uninitOuts(0);
            for (const auto& port : layer->getOutputPorts()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              for (const auto& port : layer->getOutputPorts()) {" << std::endl;
                if (port.shape().empty()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:                  if (port.shape().empty()) {" << std::endl;
                    sizes.push_back(0);
                    uninitOuts++;
                } else if (port.shape().size() <= splitLayer.getAxis()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:                  } else if (port.shape().size() <= splitLayer.getAxis()) {" << std::endl;
                    THROW_IE_EXCEPTION << "Incorrect output shapes in Split layer " << layer->getName();
                } else {
                    sizes.push_back(port.shape()[splitLayer.getAxis()]);
                    axisSize -= port.shape()[splitLayer.getAxis()];
                }
            }

            if ((axisSize && !uninitOuts) || (axisSize && uninitOuts && axisSize % uninitOuts))
                THROW_IE_EXCEPTION << "Incorrect output shapes in Split layer " << layer->getName();

            size_t commonSize = uninitOuts != 0 ? axisSize / uninitOuts : 0;
            for (size_t i = 0; i < sizes.size() && commonSize; i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              for (size_t i = 0; i < sizes.size() && commonSize; i++) {" << std::endl;
                if (!sizes[i]) sizes[i] = commonSize;
            }

            std::string out_sizes;
            for (const auto& size : sizes) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              for (const auto& size : sizes) {" << std::endl;
                if (!out_sizes.empty()) out_sizes += ",";
                out_sizes += std::to_string(size);
            }
            if (!out_sizes.empty()) params["out_sizes"] = out_sizes;
        }

        preparedParams[layer->getId()] = params;
    }

    // Try to propagate shapes
    for (auto& layer : propagatedNetwork) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto& layer : propagatedNetwork) {" << std::endl;
        // constant layer does not change during the shape inference and also the Const blob always has C layout and
        // doesn't know its real shape, so don't run shape propagation for it
        if (details::CaselessEq<std::string>()(layer->getType(), "Const")) continue;
        const auto impl = network->getContext().getShapeInferImpl(layer->getType());
        if (!impl)
            return DescriptionBuffer(NOT_FOUND, resp)
                   << "Cannot infer shapes! Shape infer implementation was not found for type " << layer->getType()
                   << ".";
        std::vector<SizeVector> inShapes;
        std::vector<SizeVector> outShapes;
        std::map<std::string, std::string> params;
        std::map<std::string, Blob::Ptr> blobs;

        std::vector<Blob::CPtr> inBlobs;
        for (const auto& inPort : layer->getInputPorts().empty() ? layer->getOutputPorts() : layer->getInputPorts()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (const auto& inPort : layer->getInputPorts().empty() ? layer->getOutputPorts() : layer->getInputPorts()) {" << std::endl;
            if (inPort.getParameters().find("type") == inPort.getParameters().end()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              if (inPort.getParameters().find('type') == inPort.getParameters().end()) {" << std::endl;
                inBlobs.push_back(inPort.getData()->getData());
            }
        }
        params = preparedParams[layer->getId()];

        for (const auto& port : layer->getInputPorts()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (const auto& port : layer->getInputPorts()) {" << std::endl;
            if (port.getParameters().find("type") == port.getParameters().end() ||
                port.getData()->getData()->cbuffer() == nullptr)
                continue;
            blobs[port.getParameters().at("type")] = port.getData()->getData();
        }
        for (const auto& it : layer->getParameters()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (const auto& it : layer->getParameters()) {" << std::endl;
            if (!it.second.is<Blob::CPtr>()) continue;
            blobs[it.first] = std::const_pointer_cast<Blob>(it.second.as<Blob::CPtr>());
        }

        StatusCode sts = impl->inferShapes(inBlobs, params, blobs, outShapes, resp);
        if (sts != OK) return sts;

        if (outShapes.size() != layer->getOutputPorts().size())
            return DescriptionBuffer(GENERAL_ERROR, resp) << "Cannot infer shapes! The number of output shapes is not "
                                                             "equal the number of output ports for layer "
                                                          << layer->getName();

        for (size_t i = 0; i < outShapes.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (size_t i = 0; i < outShapes.size(); i++) {" << std::endl;
            layer->getOutputPorts()[i].setShape(outShapes[i]);
        }
        for (const auto& connection : propagatedNetwork.getLayerConnections(layer->getId())) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (const auto& connection : propagatedNetwork.getLayerConnections(layer->getId())) {" << std::endl;
            if (connection.from().layerId() != layer->getId()) continue;
            auto nextLayer = propagatedNetwork.getLayer(connection.to().layerId());
            nextLayer->getInputPorts()[connection.to().portId()].setShape(outShapes[connection.from().portId()]);
        }
    }

    // Apply new shapes
    for (auto& layer : *network) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (auto& layer : *network) {" << std::endl;
        const auto& propagatedLayer = propagatedNetwork.getLayer(layer->getId());
        for (size_t i = 0; i < layer->getInputPorts().size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (size_t i = 0; i < layer->getInputPorts().size(); i++) {" << std::endl;
            layer->getInputPorts()[i].setShape(propagatedLayer->getInputPorts()[i].shape());
        }
        for (size_t i = 0; i < layer->getOutputPorts().size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          for (size_t i = 0; i < layer->getOutputPorts().size(); i++) {" << std::endl;
            layer->getOutputPorts()[i].setShape(propagatedLayer->getOutputPorts()[i].shape());
        }
    }

    IE_SUPPRESS_DEPRECATED_END

    return OK;
}

caseless_set<std::string> Reshaper::getTypeNamesFromExtension(const IShapeInferExtensionPtr& extension) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:  caseless_set<std::string> Reshaper::getTypeNamesFromExtension(const IShapeInferExtensionPtr& extension) {" << std::endl;
    char** types = nullptr;
    unsigned int size = 0;
    ResponseDesc resp;
    StatusCode sts = extension->getShapeInferTypes(types, size, &resp);
    if (sts != OK) THROW_IE_EXCEPTION << "Failed to get types from extension: " << resp.msg;
    caseless_set<std::string> typesSet;
    for (int i = 0; i < size; i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (int i = 0; i < size; i++) {" << std::endl;
        std::string type(types[i], strlen(types[i]));
        delete[] types[i];
        typesSet.insert(type);
    }
    delete[] types;
    return typesSet;
}

ReshapeLauncher::Ptr LauncherCreator::createNotInputLauncher(const CNNLayer* layer,
                                                             const std::vector<IShapeInferExtensionPtr>& extensions) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:                                                               const std::vector<IShapeInferExtensionPtr>& extensions) {" << std::endl;
    auto layerType = layer->type;
    if ((::details::equal(layerType, "memory") && layer->GetParamAsInt("index")) ||
        ::details::equal(layerType, "const") || ::details::equal(layerType, "input")) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          ::details::equal(layerType, 'const') || ::details::equal(layerType, 'input')) {" << std::endl;
        THROW_IE_EXCEPTION << "Failed to reshape: Layer with type `" << layerType
                           << "` can't be intermediate layer in network";
    }

    for (const auto& extension : extensions) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      for (const auto& extension : extensions) {" << std::endl;
        IShapeInferImpl::Ptr impl = nullptr;
        StatusCode sts = extension->getShapeInferImpl(impl, layerType.c_str(), nullptr);
        if (sts == OK && impl != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:          if (sts == OK && impl != nullptr) {" << std::endl;
            if (::details::equal(layerType, "memory") && !layer->GetParamAsInt("index")) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:              if (::details::equal(layerType, 'memory') && !layer->GetParamAsInt('index')) {" << std::endl;
                return std::make_shared<OutMemoryReshapeLauncher>(layer, nullptr);
            }
            return std::make_shared<ReshapeLauncher>(layer, impl);
        }
    }
    return std::make_shared<FakeReshapeLauncher>(layer, nullptr);
}

ReshapeLauncher::Ptr LauncherCreator::createInputLauncher(const CNNLayer* layer,
                                                          const std::vector<IShapeInferExtensionPtr>& extensions) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:                                                            const std::vector<IShapeInferExtensionPtr>& extensions) {" << std::endl;
    auto layerType = layer->type;
    if (::details::equal(layerType, "memory") && layer->GetParamAsInt("index")) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      if (::details::equal(layerType, 'memory') && layer->GetParamAsInt('index')) {" << std::endl;
        return std::make_shared<InputReshapeLauncher>(layer, nullptr);
    } else if (::details::equal(layerType, "const")) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      } else if (::details::equal(layerType, 'const')) {" << std::endl;
        return std::make_shared<ConstReshapeLauncher>(layer, nullptr);
    } else if (::details::equal(layerType, "input")) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshaper.cpp:      } else if (::details::equal(layerType, 'input')) {" << std::endl;
        return std::make_shared<InputReshapeLauncher>(layer, nullptr);
    }
    THROW_IE_EXCEPTION << "Failed to reshape: Layer with type `" << layerType
                       << "` can't be input. Supported input types: Input, Const and Memory(with index=1)";
}
