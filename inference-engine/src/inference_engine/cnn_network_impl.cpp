#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_impl.hpp"

#include <ie_common.h>
#include <math.h>

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <shape_infer/ie_reshaper.hpp>
#include <string>
#include <vector>
#include <unordered_set>

#include "debug.h"
#include "graph_tools.hpp"
#include "ie_profiling.hpp"
#include "network_serializer.h"
#include "details/ie_cnn_network_tools.h"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

std::map<CNNLayer*, bool> getConstLayersMap(const ICNNNetwork& network) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  std::map<CNNLayer*, bool> getConstLayersMap(const ICNNNetwork& network) {" << std::endl;
    std::map<CNNLayer*, bool> result;

    const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);
    for (const CNNLayerPtr& layer : layers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      for (const CNNLayerPtr& layer : layers) {" << std::endl;
        if (equal(layer->type, "Input") || equal(layer->type, "Memory")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          if (equal(layer->type, 'Input') || equal(layer->type, 'Memory')) {" << std::endl;
            result.emplace(layer.get(), false);
            continue;
        }

        if ((equal(layer->type, "Const")) || (equal(layer->type, "Shape")) || (equal(layer->type, "PriorBox"))) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          if ((equal(layer->type, 'Const')) || (equal(layer->type, 'Shape')) || (equal(layer->type, 'PriorBox'))) {" << std::endl;
            result.emplace(layer.get(), true);
            continue;
        }

        bool allParentsAreConst = true;
        for (const DataWeakPtr& insWeakData : layer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          for (const DataWeakPtr& insWeakData : layer->insData) {" << std::endl;
            const DataPtr insData = insWeakData.lock();
            if (insData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              if (insData == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "input data is absent";
            }

            const CNNLayerWeakPtr parentWeak = insData->getCreatorLayer();
            const CNNLayerPtr parent = parentWeak.lock();
            if (parent == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              if (parent == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "parentLayer is absent";
            }

            const auto parentIt = result.find(parent.get());
            if (parentIt == result.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              if (parentIt == result.end()) {" << std::endl;
                THROW_IE_EXCEPTION << "parent layer '" << parent->name << "' was not found";
            }

            if (!parentIt->second) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              if (!parentIt->second) {" << std::endl;
                result.emplace(layer.get(), false);
                allParentsAreConst = false;
                break;
            }
        }

        if (allParentsAreConst) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          if (allParentsAreConst) {" << std::endl;
            result.emplace(layer.get(), true);
        }
    }

    return result;
}

ICNNNetwork::~ICNNNetwork() {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  ICNNNetwork::~ICNNNetwork() {" << std::endl;}

CNNNetworkImpl::CNNNetworkImpl(): _stats(new CNNNetworkStatsImpl()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  CNNNetworkImpl::CNNNetworkImpl(): _stats(new CNNNetworkStatsImpl()) {" << std::endl;}

CNNNetworkImpl::~CNNNetworkImpl() {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  CNNNetworkImpl::~CNNNetworkImpl() {" << std::endl;
    // In case of cycles, memory leaks occur: Layer holds shared_ptr<Data>, and vice versa.
    // Added additional check on cycles.
    bool res = false;
    try {
        res = CNNNetForestDFS(CNNNetGetAllInputLayers(*this), [&](CNNLayerPtr layer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          res = CNNNetForestDFS(CNNNetGetAllInputLayers(*this), [&](CNNLayerPtr layer) {" << std::endl;}, false);
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (...) {" << std::endl;
        // Exception means that network was invalid. Reset all data.
    }

    // workaround due to memory leaks
    if (!res) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (!res) {" << std::endl;
        for (const auto& data : _data) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          for (const auto& data : _data) {" << std::endl;
            if (!data.second) continue;
            for (auto& input : data.second->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              for (auto& input : data.second->getInputTo()) {" << std::endl;
                if (!input.second) continue;
                input.second.reset();
            }
        }
    }
}

void CNNNetworkImpl::getOutputsInfo(std::map<std::string, DataPtr>& out) const noexcept {
    out = _outputData;
}

void CNNNetworkImpl::getInputsInfo(InputsDataMap& inputs) const noexcept {
    inputs = _inputData;
}

void CNNNetworkImpl::addLayer(const CNNLayerPtr& layer) noexcept {
    if (!layer) return;
    _layers[layer->name] = layer;
}

void CNNNetworkImpl::removeLayer(const std::string& layerName) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  void CNNNetworkImpl::removeLayer(const std::string& layerName) {" << std::endl;
    auto it = _layers.find(layerName);
    if (it != _layers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (it != _layers.end()) {" << std::endl;
        _layers.erase(it);
    }
}

void CNNNetworkImpl::renameLayer(const std::string& currentName, const std::string& newName) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  void CNNNetworkImpl::renameLayer(const std::string& currentName, const std::string& newName) {" << std::endl;
    const auto currentIt = _layers.find(currentName);
    if (currentIt == _layers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (currentIt == _layers.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer '" << currentName << "' was not found in layers";
    }

    if (_layers.find(newName) != _layers.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (_layers.find(newName) != _layers.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer with name '" << currentName << "' already exists in layers";
    }

    if (_inputData.find(newName) != _inputData.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (_inputData.find(newName) != _inputData.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer with name '" << currentName << "' already exists in input data";
    }

    if (_outputData.find(newName) != _outputData.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (_outputData.find(newName) != _outputData.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer with name '" << currentName << "' already exists in output data";
    }

    const auto currentDataIt = _data.find(currentName);
    if (currentDataIt == _data.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (currentDataIt == _data.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer '" << currentName << "' was not found in data";
    }

    if (_data.find(newName) != _data.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (_data.find(newName) != _data.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer with name '" << currentName << "' already exists in data";
    }

    bool wasUpdatedInput = false;
    for (auto inputDataIt = _inputData.begin(); inputDataIt != _inputData.end(); ++inputDataIt) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      for (auto inputDataIt = _inputData.begin(); inputDataIt != _inputData.end(); ++inputDataIt) {" << std::endl;
        const CNNLayerPtr inputLayer = inputDataIt->second->getInputData()->getCreatorLayer().lock();
        if (inputLayer->name == currentName) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          if (inputLayer->name == currentName) {" << std::endl;
            _inputData.emplace(newName, inputDataIt->second);
            _inputData.erase(inputDataIt);
            wasUpdatedInput = true;
            break;
        }
    }

    if (!wasUpdatedInput) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (!wasUpdatedInput) {" << std::endl;
        for (auto outputDataIt = _outputData.begin(); outputDataIt != _outputData.end(); ++outputDataIt) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          for (auto outputDataIt = _outputData.begin(); outputDataIt != _outputData.end(); ++outputDataIt) {" << std::endl;
            const CNNLayerPtr outputLayer = outputDataIt->second->getCreatorLayer().lock();
            if (outputLayer->name == currentName) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              if (outputLayer->name == currentName) {" << std::endl;
                _outputData.emplace(newName, outputDataIt->second);
                _outputData.erase(outputDataIt);
                break;
            }
        }
    }

    _layers.emplace(newName, currentIt->second);
    currentIt->second->name = newName;
    _layers.erase(currentIt);

    _data.emplace(newName, currentDataIt->second);
    currentDataIt->second->setName(newName);
    _data.erase(currentDataIt);
}

void CNNNetworkImpl::removeData(const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  void CNNNetworkImpl::removeData(const std::string& dataName) {" << std::endl;
    auto it = _data.find(dataName);
    if (it != _data.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (it != _data.end()) {" << std::endl;
        _data.erase(it);
    }
}

void CNNNetworkImpl::validate(int version) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  void CNNNetworkImpl::validate(int version) {" << std::endl;
    std::set<std::string> layerNames;
    std::set<std::string> dataNames;

    InputsDataMap inputs;
    this->getInputsInfo(inputs);
    if (inputs.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (inputs.empty()) {" << std::endl;
        THROW_IE_EXCEPTION << "No input layers";
    }

    bool res = CNNNetForestDFS(
        CNNNetGetAllInputLayers(*this),
        [&](CNNLayerPtr layer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          [&](CNNLayerPtr layer) {" << std::endl;
            std::string layerName = layer->name;

            for (auto i : layer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              for (auto i : layer->insData) {" << std::endl;
                auto data = i.lock();
                if (data) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                  if (data) {" << std::endl;
                    auto inputTo = data->getInputTo();
                    auto iter = inputTo.find(layerName);
                    auto dataName = data->getName();
                    if (iter == inputTo.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                      if (iter == inputTo.end()) {" << std::endl;
                        THROW_IE_EXCEPTION << "Data " << data->getName() << " which inserted into the layer "
                                           << layerName << " does not point at this layer";
                    }
                    if (!data->getCreatorLayer().lock()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                      if (!data->getCreatorLayer().lock()) {" << std::endl;
                        THROW_IE_EXCEPTION << "Data " << dataName << " has no creator layer";
                    }
                } else {
                    THROW_IE_EXCEPTION << "Data which inserted into the layer " << layerName << " is nullptr";
                }
            }
            for (auto data : layer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              for (auto data : layer->outData) {" << std::endl;
                auto inputTo = data->getInputTo();
                std::string dataName = data->getName();
                for (auto layerIter : inputTo) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                  for (auto layerIter : inputTo) {" << std::endl;
                    CNNLayerPtr layerInData = layerIter.second;
                    if (!layerInData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                      if (!layerInData) {" << std::endl;
                        THROW_IE_EXCEPTION << "Layer which takes data " << dataName << " is nullptr";
                    }
                    auto insertedDatas = layerInData->insData;

                    auto it =
                        std::find_if(insertedDatas.begin(), insertedDatas.end(), [&](InferenceEngine::DataWeakPtr& d) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                          std::find_if(insertedDatas.begin(), insertedDatas.end(), [&](InferenceEngine::DataWeakPtr& d) {" << std::endl;
                            return d.lock() == data;
                        });
                    if (it == insertedDatas.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                      if (it == insertedDatas.end()) {" << std::endl;
                        THROW_IE_EXCEPTION << "Layer " << layerInData->name << " which takes data " << dataName
                                           << " does not point at this data";
                    }
                }
                auto dataNameSetPair = dataNames.insert(dataName);
                if (!dataNameSetPair.second) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                  if (!dataNameSetPair.second) {" << std::endl;
                    THROW_IE_EXCEPTION << "Data name " << dataName << " is not unique";
                }
            }
            auto layerSetPair = layerNames.insert(layerName);
            if (!layerSetPair.second) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              if (!layerSetPair.second) {" << std::endl;
                THROW_IE_EXCEPTION << "Layer name " << layerName << " is not unique";
            }
        },
        false);

    std::string inputType = "Input";
    for (auto i : inputs) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      for (auto i : inputs) {" << std::endl;
        CNNLayerPtr layer = i.second->getInputData()->getCreatorLayer().lock();
        if (layer && !equal(layer->type, inputType)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          if (layer && !equal(layer->type, inputType)) {" << std::endl;
            THROW_IE_EXCEPTION << "Input layer " << layer->name << " should have Input type but actually its type is "
                               << layer->type;
        }
    }

    if (!res) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (!res) {" << std::endl;
        THROW_IE_EXCEPTION << "Sorting not possible, due to existed loop.";
    }
}

StatusCode CNNNetworkImpl::getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept {
    auto it = _layers.find(layerName);
    if (it == _layers.end())
        return DescriptionBuffer(NOT_FOUND, resp) << "Layer " << layerName << " not found in network";
    out = it->second;
    return OK;
}

StatusCode CNNNetworkImpl::addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept {
    CNNLayerPtr outLayer;
    auto rc = getLayerByName(layerName.c_str(), outLayer, resp);
    if (rc != OK) return rc;

    if (outputIndex >= outLayer->outData.size())
        return DescriptionBuffer(OUT_OF_BOUNDS, resp)
               << "port index " << outputIndex << " exceeds layer's outputs which is " << outLayer->outData.size();
    shared_ptr<Data> outData = outLayer->outData[outputIndex];
    _outputData[outData->getName()] = outData;
    return OK;
}

void CNNNetworkImpl::resolveOutput() {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  void CNNNetworkImpl::resolveOutput() {" << std::endl;
    // check orphan nodes...
    for (auto kvp : _data) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      for (auto kvp : _data) {" << std::endl;
        if (!kvp.second->isInitialized())
            THROW_IE_EXCEPTION << "data name [" << kvp.first << "] dimensions is not known";

        // data nodes not going to any layer are basically graph output...
        if (kvp.second->getInputTo().empty()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          if (kvp.second->getInputTo().empty()) {" << std::endl;
            _outputData[kvp.first] = kvp.second;
        }
    }
}

void CNNNetworkImpl::addOutput(const string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  void CNNNetworkImpl::addOutput(const string& dataName) {" << std::endl;
    auto it = _data.find(dataName);
    if (it == _data.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (it == _data.end()) {" << std::endl;
        THROW_IE_EXCEPTION << "data [" << dataName << "] doesn't exist";
    }
    auto data = it->second;
    assert(data->getName() == dataName);
    _outputData[dataName] = data;
}

void CNNNetworkImpl::removeOutput(const string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:  void CNNNetworkImpl::removeOutput(const string& dataName) {" << std::endl;
    removeData(dataName);

    auto it = _outputData.find(dataName);
    if (it != _outputData.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (it != _outputData.end()) {" << std::endl;
        _outputData.erase(it);
    }
}

size_t CNNNetworkImpl::getBatchSize() const noexcept {
    if (!_inputData.size()) return 0;
    // currently CNNNetworkImpl::setBatchSize set the same values
    // for the latest dim as a batch, we can take the first input
    // and return batch size for it
    SizeVector dims = _inputData.cbegin()->second->getTensorDesc().getDims();
    // 3D input layout doesn't have batch notation for input so batch is 1
    if (dims.size() == 3 || dims.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      if (dims.size() == 3 || dims.size() == 1) {" << std::endl;
        return 1;
    }
    return dims.at(0);
}

StatusCode CNNNetworkImpl::reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                                   ResponseDesc* responseDesc) noexcept {
    try {
        if (!_reshaper) _reshaper = std::make_shared<ShapeInfer::Reshaper>(*this);
        _reshaper->run(inputShapes);
    } catch (const InferenceEngineException& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const InferenceEngineException& e) {" << std::endl;
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << e.what();
    } catch (const std::exception& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const std::exception& e) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, responseDesc) << e.what();
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (...) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, responseDesc);
    }
    return OK;
}

StatusCode CNNNetworkImpl::AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension,
                                        InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        if (!_reshaper) _reshaper = std::make_shared<ShapeInfer::Reshaper>(*this);
        _reshaper->AddExtension(extension);
    } catch (const InferenceEngineException& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const InferenceEngineException& e) {" << std::endl;
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    } catch (const std::exception& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const std::exception& e) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, resp) << e.what();
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (...) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, resp);
    }
    return OK;
}

StatusCode CNNNetworkImpl::serialize(const std::string& xmlPath, const std::string& binPath, ResponseDesc* resp) const
    noexcept {
    try {
        NetworkSerializer::serialize(xmlPath, binPath, (InferenceEngine::ICNNNetwork&)*this);
    } catch (const InferenceEngineException& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const InferenceEngineException& e) {" << std::endl;
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    } catch (const std::exception& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const std::exception& e) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, resp) << e.what();
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (...) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, resp);
    }
    return OK;
}

StatusCode CNNNetworkImpl::setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept {
    try {
        auto originalBatchSize = getBatchSize();
        if (originalBatchSize == size) return OK;
        SizeVector dims = _inputData.cbegin()->second->getTensorDesc().getDims();

        // 3D input layout doesn't have batch notation
        if (dims.size() == 3 || dims.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          if (dims.size() == 3 || dims.size() == 1) {" << std::endl;
            return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc) << "Cannot set batch for 1D/3D input";
        }

        const std::map<CNNLayer*, bool> layersMap = getConstLayersMap(*this);
        for (auto& layer : _data) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          for (auto& layer : _data) {" << std::endl;
            SizeVector dims = layer.second->getDims();
            CNNLayerPtr layerT = layer.second->getCreatorLayer().lock();

            bool constOrAbsent;
            if (layerT) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              if (layerT) {" << std::endl;
                const auto it = layersMap.find(layerT.get());
                if (it == layersMap.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                  if (it == layersMap.end()) {" << std::endl;
                    THROW_IE_EXCEPTION << "layer '" << layerT->name << "' was not found in layers map";
                }
                constOrAbsent = it->second;
            } else {
                constOrAbsent = true;
            }

            if (!layerT || !constOrAbsent) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              if (!layerT || !constOrAbsent) {" << std::endl;
                float diff = static_cast<float>(dims.at(0)) / static_cast<float>(originalBatchSize);
                dims.at(0) = static_cast<size_t>(std::ceil(size * diff));
                layer.second->setDims(dims);
            }
        }
        return OK;
    } catch (const InferenceEngineException& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const InferenceEngineException& e) {" << std::endl;
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << e.what();
    } catch (const std::exception& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const std::exception& e) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, responseDesc) << e.what();
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (...) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, responseDesc);
    }
}

StatusCode CNNNetworkImpl::setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept {
    InputShapes inputShapes;
    try {
        for (const auto& pair : _inputData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:          for (const auto& pair : _inputData) {" << std::endl;
            auto info = pair.second;
            if (info) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:              if (info) {" << std::endl;
                auto data = info->getInputData();
                if (data) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:                  if (data) {" << std::endl;
                    auto dims = data->getTensorDesc().getDims();
                    dims[0] = size;
                    inputShapes[data->getName()] = dims;
                }
            }
        }
        return reshape(inputShapes, responseDesc);
    } catch (const InferenceEngineException& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const InferenceEngineException& e) {" << std::endl;
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << e.what();
    } catch (const std::exception& e) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (const std::exception& e) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, responseDesc) << e.what();
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_impl.cpp:      } catch (...) {" << std::endl;
        return DescriptionBuffer(UNEXPECTED, responseDesc);
    }
}
