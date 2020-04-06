#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_infer/ie_reshape_io_controllers.hpp"

#include <ie_layers.h>

#include <blob_factory.hpp>
#include <ie_layer_validators.hpp>
#include <set>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace ShapeInfer;

void DefaultChecker::run(const std::vector<DataPtr>& dataVec, const std::string& layerName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void DefaultChecker::run(const std::vector<DataPtr>& dataVec, const std::string& layerName) {" << std::endl;
    std::string errorBase = "Failed to init controller for reshaping layer `" + layerName + "`";
    if (dataVec.empty()) THROW_IE_EXCEPTION << errorBase + ": vector of data is empty";
    for (const auto& data : dataVec) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      for (const auto& data : dataVec) {" << std::endl;
        if (!data) THROW_IE_EXCEPTION << errorBase + ": pointer to the data is null";
    }
}

InputController::InputController(const std::vector<DataPtr>& dataVec, const std::string& layerName,
                                 const DefaultChecker::Ptr& checker)
    : _dataVec(dataVec), _layerName(layerName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      : _dataVec(dataVec), _layerName(layerName) {" << std::endl;
    checker->run(_dataVec, layerName);
    for (const auto& data : _dataVec) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      for (const auto& data : _dataVec) {" << std::endl;
        if (data) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:          if (data) {" << std::endl;
            _dataNames.push_back(data->getName());
            SizeVector dims = data->getTensorDesc().getDims();
            _irShapes.push_back(dims);
            // TODO probably need to create blobs with dimensions, not on getBlobs stage
            _inferedData.push_back(nullptr);
        }
    }
    _shapes = _irShapes;
}

void InputController::setShapeByName(const SizeVector& shape, const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void InputController::setShapeByName(const SizeVector& shape, const std::string& dataName) {" << std::endl;
    long pos = getPositionByName(dataName);
    _shapes[pos] = shape;
}

SizeVector InputController::getShapeByName(const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  SizeVector InputController::getShapeByName(const std::string& dataName) {" << std::endl;
    long pos = getPositionByName(dataName);
    return _shapes[pos];
}

std::vector<SizeVector> InputController::getShapes(bool check) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  std::vector<SizeVector> InputController::getShapes(bool check) {" << std::endl;
    if (check) checkCorrespondence();
    return _shapes;
}

void InputController::applyChanges() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void InputController::applyChanges() {" << std::endl;
    checkCorrespondence();
    for (int i = 0; i < _dataVec.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      for (int i = 0; i < _dataVec.size(); i++) {" << std::endl;
        auto data = _dataVec[i];
        if (data) data->setDims(_shapes[i]);
    }
}

void InputController::checkCorrespondence() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void InputController::checkCorrespondence() {" << std::endl;
    if (_shapes.size() != _dataVec.size()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      if (_shapes.size() != _dataVec.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "ReshapeLauncher: Number of data(" << _dataVec.size()
                           << ") doesn't match with number of shapes(" << _shapes.size() << ") for layer '"
                           << _layerName << "'!";
    }
    // TODO: iterate and check for emptiness and size matching
}

void InputController::reset() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void InputController::reset() {" << std::endl;
    _shapes = _irShapes;
}

std::vector<SizeVector> InputController::getIRShapes() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  std::vector<SizeVector> InputController::getIRShapes() {" << std::endl;
    return _irShapes;
}

SizeVector InputController::getIRShapeByName(const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  SizeVector InputController::getIRShapeByName(const std::string& dataName) {" << std::endl;
    long pos = getPositionByName(dataName);
    return _irShapes[pos];
}

long InputController::getPositionByName(const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  long InputController::getPositionByName(const std::string& dataName) {" << std::endl;
    auto pos = std::distance(_dataNames.begin(), std::find(_dataNames.begin(), _dataNames.end(), dataName));
    if (pos < 0 || pos >= _dataNames.size()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      if (pos < 0 || pos >= _dataNames.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Failed to find shape that corresponds Data name=" << dataName;
    }
    return pos;
}

void InputController::setShapeByIndex(const SizeVector& shape, size_t index) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void InputController::setShapeByIndex(const SizeVector& shape, size_t index) {" << std::endl;
    size_t numShapes = _shapes.size();
    if (index >= numShapes) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      if (index >= numShapes) {" << std::endl;
        THROW_IE_EXCEPTION << "Failed to set shape for index(" << index
                           << ") that is more than number of shapes: " << numShapes;
    }
    _shapes[index] = shape;
}

bool InputController::isDataAvailable() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  bool InputController::isDataAvailable() {" << std::endl;
    if (_inferedData.empty()) return false;
    for (const auto& data : _inferedData) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      for (const auto& data : _inferedData) {" << std::endl;
        if (!data)
            return false;
        else if (data->cbuffer() == nullptr)
            return false;
    }
    return true;
}

std::vector<Blob::CPtr> InputController::getBlobs(bool check) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  std::vector<Blob::CPtr> InputController::getBlobs(bool check) {" << std::endl;
    if (check) checkCorrespondence();
    for (int i = 0; i < _dataVec.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      for (int i = 0; i < _dataVec.size(); i++) {" << std::endl;
        if (_inferedData[i] == nullptr || _inferedData[i]->cbuffer() == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:          if (_inferedData[i] == nullptr || _inferedData[i]->cbuffer() == nullptr) {" << std::endl;
            TensorDesc desc = _dataVec[i]->getTensorDesc();
            desc.setDims(_shapes[i]);
            // special case of Shape layer: no input data, but blob contains info about dimensions, layout and etc...
            auto blob = make_blob_with_precision(desc);
            _inferedData[i] = blob;
        }
    }
    return _inferedData;
}

void InputController::setBlobByName(const Blob::CPtr& blob, const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void InputController::setBlobByName(const Blob::CPtr& blob, const std::string& dataName) {" << std::endl;
    long pos = getPositionByName(dataName);
    _inferedData[pos] = blob;
}

OutputController::OutputController(const std::vector<DataPtr>& data, const std::string& layerName,
                                   const DefaultChecker::Ptr& checker)
    : InputController(data, layerName, checker) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      : InputController(data, layerName, checker) {" << std::endl;}

void OutputController::propagateShapes(const std::set<ReshapeLauncher::Ptr>& launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void OutputController::propagateShapes(const std::set<ReshapeLauncher::Ptr>& launchers) {" << std::endl;
    checkCorrespondence();
    unsigned idx = 0;
    for (auto const& outData : _dataVec) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      for (auto const& outData : _dataVec) {" << std::endl;
        for (auto const& inputTo : outData->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:          for (auto const& inputTo : outData->getInputTo()) {" << std::endl;
            CNNLayerPtr layer = inputTo.second;
            if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:              if (layer == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "Failed to propagate shapes for layer (" << inputTo.first
                                   << "): connected layer is null";
            }
            auto layerName = layer->name;
            auto foundLauncher =
                std::find_if(launchers.begin(), launchers.end(), [&layerName](const ReshapeLauncher::Ptr& launcher) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:                  std::find_if(launchers.begin(), launchers.end(), [&layerName](const ReshapeLauncher::Ptr& launcher) {" << std::endl;
                    return launcher->getLayerName() == layerName;
                });
            if (foundLauncher == launchers.end())
                THROW_IE_EXCEPTION << "Failed to find ReshapeLauncher for layer: '" << layerName << "'";
            (*foundLauncher)->setShapeByName(_shapes[idx], outData->getName());
        }
        idx++;
    }
}

// Combine with propagate shapes
void OutputController::propagateBlobs(const std::set<ReshapeLauncher::Ptr>& launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void OutputController::propagateBlobs(const std::set<ReshapeLauncher::Ptr>& launchers) {" << std::endl;
    unsigned idx = 0;
    for (auto const& outData : _dataVec) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      for (auto const& outData : _dataVec) {" << std::endl;
        for (auto const& inputTo : outData->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:          for (auto const& inputTo : outData->getInputTo()) {" << std::endl;
            CNNLayerPtr layer = inputTo.second;
            if (layer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:              if (layer == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "Failed to propagate shapes for layer (" << inputTo.first
                                   << "): connected layer is null";
            }
            auto layerName = layer->name;
            auto foundLauncher =
                std::find_if(launchers.begin(), launchers.end(), [&layerName](const ReshapeLauncher::Ptr& launcher) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:                  std::find_if(launchers.begin(), launchers.end(), [&layerName](const ReshapeLauncher::Ptr& launcher) {" << std::endl;
                    return launcher->getLayerName() == layerName;
                });
            if (foundLauncher == launchers.end())
                THROW_IE_EXCEPTION << "Failed to find ReshapeLauncher for layer: '" << layerName << "'";
            (*foundLauncher)->setBlobByName(_inferedData[idx], outData->getName());
        }
        idx++;
    }
}

void OutputController::setShapes(const std::vector<SizeVector>& shapes) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void OutputController::setShapes(const std::vector<SizeVector>& shapes) {" << std::endl;
    _shapes = shapes;
}

void OutputController::setBlobs(const std::vector<Blob::Ptr>& blobs) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  void OutputController::setBlobs(const std::vector<Blob::Ptr>& blobs) {" << std::endl;
    _inferedData.clear();
    for (const auto& blob : blobs) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      for (const auto& blob : blobs) {" << std::endl;
        _inferedData.push_back(blob);
    }
}

std::vector<Blob::Ptr> OutputController::createBlobs() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:  std::vector<Blob::Ptr> OutputController::createBlobs() {" << std::endl;
    std::vector<Blob::Ptr> blobs;
    for (int i = 0; i < _dataVec.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_io_controllers.cpp:      for (int i = 0; i < _dataVec.size(); i++) {" << std::endl;
        TensorDesc desc = _dataVec[i]->getTensorDesc();
        desc.setDims(_shapes[i]);
        auto blob = make_blob_with_precision(desc);
        blob->allocate();
        blobs.push_back(blob);
    }
    return blobs;
}
