#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_infer/ie_reshape_launcher.hpp"

#include <ie_layers.h>

#include <details/ie_exception.hpp>
#include <ie_layer_validators.hpp>
#include <map>
#include <memory>
#include <set>
#include <shape_infer/const_infer/ie_const_infer_holder.hpp>
#include <string>
#include <vector>

#include "built-in/ie_tensor_iterator_shape_infer.hpp"
#include "ie_reshape_launcher.hpp"
#include "shape_infer/ie_reshape_io_controllers.hpp"

using namespace InferenceEngine;
using namespace ShapeInfer;

void DefaultInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void DefaultInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {" << std::endl;
    std::string errorBase = "Failed to init reshape launcher: ";
    if (!layer) THROW_IE_EXCEPTION << errorBase + " pointer to the layer is null";
    if (!impl) THROW_IE_EXCEPTION << errorBase + " shape infer implementation is null";
}

InputController* DefaultInitializer::createInputController(const CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  InputController* DefaultInitializer::createInputController(const CNNLayer* layer) {" << std::endl;
    std::vector<DataPtr> data;
    for (auto const& insData : layer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      for (auto const& insData : layer->insData) {" << std::endl;
        data.push_back(insData.lock());
    }
    return new InputController(data, layer->name);
}

OutputController* DefaultInitializer::createOutputController(const CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  OutputController* DefaultInitializer::createOutputController(const CNNLayer* layer) {" << std::endl;
    return new OutputController(layer->outData, layer->name);
}

ReshapeLauncher::ReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl,
                                 const DefaultInitializer::Ptr& initializer)
    : _layer(layer), _reshapeImpl(impl) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      : _layer(layer), _reshapeImpl(impl) {" << std::endl;
    initializer->check(layer, impl);
    ConstInferHolder holder;
    if (layer) _inferImpl = holder.getConstInferImpl(layer->type);
    try {
        _iController = initializer->createInputController(layer);
        _oController = initializer->createOutputController(layer);
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      } catch (...) {" << std::endl;
        auto exception = std::current_exception();
        delete _iController;
        delete _oController;
        std::rethrow_exception(exception);
    }
}

ReshapeLauncher::~ReshapeLauncher() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  ReshapeLauncher::~ReshapeLauncher() {" << std::endl;
    delete _iController;
    delete _oController;
    _iController = nullptr;
    _oController = nullptr;
}

void ReshapeLauncher::setShapeByName(const SizeVector& shape, const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ReshapeLauncher::setShapeByName(const SizeVector& shape, const std::string& dataName) {" << std::endl;
    _iController->setShapeByName(shape, dataName);
}

void ReshapeLauncher::setBlobByName(const Blob::CPtr& blob, const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ReshapeLauncher::setBlobByName(const Blob::CPtr& blob, const std::string& dataName) {" << std::endl;
    _iController->setBlobByName(blob, dataName);
}

SizeVector ReshapeLauncher::getShapeByName(const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  SizeVector ReshapeLauncher::getShapeByName(const std::string& dataName) {" << std::endl;
    return _oController->getShapeByName(dataName);
}

void ReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {" << std::endl;
    ResponseDesc resp;
    std::vector<SizeVector> outShapes;

    // TODO: TensorIterator strongly required original layer instance because body is not presented
    //       in params map. Original subnetwork body is required for internal shape infer
    TensorIteratorShapeProp* TI_shaper = dynamic_cast<TensorIteratorShapeProp*>(_reshapeImpl.get());
    if (TI_shaper) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      if (TI_shaper) {" << std::endl;
        TI_shaper->setOriginalLayer(_layer);
    }

    auto sts = _reshapeImpl->inferShapes(_iController->getBlobs(true), _layer->params, _layer->blobs, outShapes, &resp);
    _oController->setShapes(outShapes);
    if (sts != OK)
        THROW_IE_EXCEPTION << "Failed to infer shapes for " + _layer->type + " layer (" + _layer->name +
                                  ") with error: " + resp.msg;
    _oController->propagateShapes(launchers);
}

void ReshapeLauncher::applyChanges(CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ReshapeLauncher::applyChanges(CNNLayer* layer) {" << std::endl;
    checkLayer(layer);
    _iController->applyChanges();
    _oController->applyChanges();

    // TODO: Need to finalize result of internal body shape infer and apply
    //       new shapes to body subnetwork
    TensorIteratorShapeProp* TI_shaper = dynamic_cast<TensorIteratorShapeProp*>(_reshapeImpl.get());
    if (TI_shaper) TI_shaper->apply();
}

void ReshapeLauncher::constInfer(const std::set<ReshapeLauncher::Ptr>& launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ReshapeLauncher::constInfer(const std::set<ReshapeLauncher::Ptr>& launchers) {" << std::endl;
    if ((_iController->isDataAvailable() && _layer->type != "Quantize" && _layer->type != "FakeQuantize") ||
        _layer->type == "Const" || _layer->type == "Shape") {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:          _layer->type == 'Const' || _layer->type == 'Shape') {" << std::endl;
        auto outBlobs = _oController->createBlobs();
        _oController->setBlobs(outBlobs);
        if (!_inferImpl)
            THROW_IE_EXCEPTION << "Failed to find reference implementation for `" + _layer->name + "` Layer with `" +
                                      _layer->type + "` Type on constant propagation";
        _inferImpl->infer(_iController->getBlobs(false), _layer->params, _layer->blobs, outBlobs);
        _oController->propagateBlobs(launchers);
    }
}

void ReshapeLauncher::reset() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ReshapeLauncher::reset() {" << std::endl;
    _iController->reset();
    _oController->reset();
}

std::string ReshapeLauncher::getLayerName() const {
    return _layer->name;
}

std::string ReshapeLauncher::getLayerType() const {
    return _layer->type;
}

void ReshapeLauncher::checkLayer(CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ReshapeLauncher::checkLayer(CNNLayer* layer) {" << std::endl;
    if ((nullptr == _layer || layer == nullptr)) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      if ((nullptr == _layer || layer == nullptr)) {" << std::endl;
        THROW_IE_EXCEPTION << "Can't apply changes for empty layer";
    }
    auto oldParams = _layer->params;
    auto newParams = layer->params;
    if ((!oldParams.empty() && !newParams.empty() &&
         !std::equal(oldParams.begin(), oldParams.end(), newParams.begin())) ||
        (_layer->name != layer->name) || (_layer->type != layer->type) || oldParams.size() != newParams.size()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:          (_layer->name != layer->name) || (_layer->type != layer->type) || oldParams.size() != newParams.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Can't apply changes for layer with another params";
    }
}

void ReshapeLauncher::setIRShapeByName(const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ReshapeLauncher::setIRShapeByName(const std::string& dataName) {" << std::endl;
    SizeVector foundShape = _iController->getIRShapeByName(dataName);
    _iController->setShapeByName(foundShape, dataName);
}

void ReshapeLauncher::setShapeInferImpl(const IShapeInferImpl::Ptr& impl) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ReshapeLauncher::setShapeInferImpl(const IShapeInferImpl::Ptr& impl) {" << std::endl;
    _reshapeImpl = impl;
}

const CNNLayer* ReshapeLauncher::getLayer() const {
    return _layer;
}

InputController* FakeInitializer::createInputController(const CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  InputController* FakeInitializer::createInputController(const CNNLayer* layer) {" << std::endl;
    std::vector<DataPtr> outData;
    for (auto const& insData : layer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      for (auto const& insData : layer->insData) {" << std::endl;
        outData.push_back(insData.lock());
    }
    return new InputController(outData, layer->name);
}

void FakeInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void FakeInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {" << std::endl;
    std::string errorBase = "Failed to init reshape launcher: ";
    if (!layer) THROW_IE_EXCEPTION << errorBase + " pointer to the layer is null";
}

OutputController* FakeInitializer::createOutputController(const CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  OutputController* FakeInitializer::createOutputController(const CNNLayer* layer) {" << std::endl;
    return new OutputController(layer->outData, layer->name);
}

FakeReshapeLauncher::FakeReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl)
    : ReshapeLauncher(layer, impl, std::make_shared<FakeInitializer>()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      : ReshapeLauncher(layer, impl, std::make_shared<FakeInitializer>()) {" << std::endl;}

void FakeReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void FakeReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {" << std::endl;
    auto iShapesIR = _iController->getIRShapes();
    auto oShapesIR = _oController->getIRShapes();
    auto iShapes = _iController->getShapes(true);

    for (int i = 0; i < iShapes.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      for (int i = 0; i < iShapes.size(); i++) {" << std::endl;
        auto newInShape = iShapes[i];
        auto irInShape = iShapesIR[i];
        bool equal = std::equal(newInShape.begin(), newInShape.end(), irInShape.begin());
        if (!equal) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:          if (!equal) {" << std::endl;
            THROW_IE_EXCEPTION << "Failed to infer shapes for layer with type: " << _layer->type
                               << ". Use @IShapeInferExtension class to register shape infer function for this layer";
        }
    }

    _oController->setShapes(oShapesIR);
    _oController->propagateShapes(launchers);
}

void OutputOnlyInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutputOnlyInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {" << std::endl;
    std::string errorBase = "Failed to init reshape launcher: ";
    if (!layer) THROW_IE_EXCEPTION << errorBase + " pointer to the layer is null";
    if (!layer->insData.empty())
        THROW_IE_EXCEPTION << "Failed to init reshape launcher: "
                           << "layer type (`" + layer->type + "`) is supposed to not have inputs, but actually it has";
}

InputController* OutputOnlyInitializer::createInputController(const CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  InputController* OutputOnlyInitializer::createInputController(const CNNLayer* layer) {" << std::endl;
    return nullptr;
}

OutputController* OutputOnlyInitializer::createOutputController(const CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  OutputController* OutputOnlyInitializer::createOutputController(const CNNLayer* layer) {" << std::endl;
    return new OutputController(layer->outData, layer->name);
}

OutputOnlyReshapeLauncher::OutputOnlyReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl,
                                                     const OutputOnlyInitializer::Ptr& initializer)
    : ReshapeLauncher(layer, impl, initializer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      : ReshapeLauncher(layer, impl, initializer) {" << std::endl;}

void OutputOnlyReshapeLauncher::setShapeByName(const SizeVector& shape, const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutputOnlyReshapeLauncher::setShapeByName(const SizeVector& shape, const std::string& dataName) {" << std::endl;
    _oController->setShapeByName(shape, dataName);
}

void OutputOnlyReshapeLauncher::setBlobByName(const Blob::CPtr& blob, const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutputOnlyReshapeLauncher::setBlobByName(const Blob::CPtr& blob, const std::string& dataName) {" << std::endl;
    _oController->setBlobByName(blob, dataName);
}

void OutputOnlyReshapeLauncher::setIRShapeByName(const std::string& dataName) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutputOnlyReshapeLauncher::setIRShapeByName(const std::string& dataName) {" << std::endl;
    SizeVector foundShape = _oController->getIRShapeByName(dataName);
    _oController->setShapeByName(foundShape, dataName);
}

void OutputOnlyReshapeLauncher::applyChanges(CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutputOnlyReshapeLauncher::applyChanges(CNNLayer* layer) {" << std::endl;
    checkLayer(layer);
    _oController->applyChanges();
}

void OutputOnlyReshapeLauncher::reset() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutputOnlyReshapeLauncher::reset() {" << std::endl;
    _oController->reset();
}

void OutputOnlyReshapeLauncher::constInfer(const std::set<ReshapeLauncher::Ptr>& launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutputOnlyReshapeLauncher::constInfer(const std::set<ReshapeLauncher::Ptr>& launchers) {" << std::endl;
    if (_layer->type == "Const") {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      if (_layer->type == 'Const') {" << std::endl;
        auto outBlobs = _oController->createBlobs();
        _oController->setBlobs(outBlobs);
        if (!_inferImpl)
            THROW_IE_EXCEPTION << "Failed to find reference implementation for `" + _layer->name + "` Layer with `" +
                                      _layer->type + "` Type on constant propagation";
        _inferImpl->infer({}, _layer->params, _layer->blobs, outBlobs);
        auto shapes = _oController->getShapes(true);
        for (int i = 0; i < outBlobs.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:          for (int i = 0; i < outBlobs.size(); i++) {" << std::endl;
            outBlobs[i]->getTensorDesc().reshape(shapes[i], TensorDesc::getLayoutByDims(shapes[i]));
        }
        _oController->setBlobs(outBlobs);
        _oController->propagateBlobs(launchers);
    }
}

void InputInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void InputInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {" << std::endl;
    OutputOnlyInitializer::check(layer, impl);
    std::string errorBase = "Failed to init reshape launcher: layer type (`" + layer->type + "`) is not";
    if (details::equal(layer->type, "memory")) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      if (details::equal(layer->type, 'memory')) {" << std::endl;
        if (!layer->GetParamAsInt("index")) THROW_IE_EXCEPTION << errorBase << " `Memory`(as input)";
    } else if (!::details::equal(layer->type, "input")) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      } else if (!::details::equal(layer->type, 'input')) {" << std::endl;
        THROW_IE_EXCEPTION << errorBase << " `Input`";
    }
}

InputReshapeLauncher::InputReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl,
                                           const DefaultInitializer::Ptr& initializer)
    : OutputOnlyReshapeLauncher(layer, impl, initializer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      : OutputOnlyReshapeLauncher(layer, impl, initializer) {" << std::endl;}

void InputReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void InputReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {" << std::endl;
    auto oShapes = _oController->getShapes(false);
    auto oIRShapes = _oController->getIRShapes();
    for (size_t i = 0; i < oShapes.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      for (size_t i = 0; i < oShapes.size(); i++) {" << std::endl;
        if (oShapes[i].empty()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:          if (oShapes[i].empty()) {" << std::endl;
            _oController->setShapeByIndex(oIRShapes[i], i);
        }
    }
    _oController->propagateShapes(launchers);
}

void ConstInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ConstInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {" << std::endl;
    OutputOnlyInitializer::check(layer, impl);
    if (!::details::equal(layer->type, "const"))
        THROW_IE_EXCEPTION << "Failed to init reshape launcher: layer type (`" + layer->type + "`) is not `Const`";
}

ConstReshapeLauncher::ConstReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl)
    : OutputOnlyReshapeLauncher(layer, impl, std::make_shared<ConstInitializer>()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      : OutputOnlyReshapeLauncher(layer, impl, std::make_shared<ConstInitializer>()) {" << std::endl;}

void ConstReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void ConstReshapeLauncher::reshape(const std::set<ReshapeLauncher::Ptr>& launchers) {" << std::endl;
    auto oShapesIR = _oController->getIRShapes();
    auto oShapes = _oController->getShapes(false);

    if (oShapes.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      if (oShapes.empty()) {" << std::endl;
        _oController->setShapes(oShapesIR);
    }
    if (oShapes != oShapesIR) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      if (oShapes != oShapesIR) {" << std::endl;
        THROW_IE_EXCEPTION << "Failed to set different shapes for Const layer,"
                           << " original shapes:" << details::dumpVec(oShapesIR)
                           << " new shapes:" << details::dumpVec(oShapes);
    }
    _oController->propagateShapes(launchers);
}

void OutMemoryInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutMemoryInitializer::check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) {" << std::endl;
    std::string errorBase = "Failed to init reshape launcher: ";
    if (!layer) THROW_IE_EXCEPTION << errorBase + " pointer to the layer is null";
    int index = layer->GetParamAsInt("index");
    if (!::details::equal(layer->type, "memory") && index)
        THROW_IE_EXCEPTION << "Failed to init reshape launcher: layer type (`" + layer->type +
                                  "`) is not `Memory` as output";
    if (!layer->outData.empty())
        THROW_IE_EXCEPTION << "Failed to init reshape launcher: "
                           << "layer type (`" + layer->type + "`) is supposed to not have outputs, but actually it has";
}

OutputController* OutMemoryInitializer::createOutputController(const CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  OutputController* OutMemoryInitializer::createOutputController(const CNNLayer* layer) {" << std::endl;
    return nullptr;
}

OutMemoryReshapeLauncher::OutMemoryReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl)
    : ReshapeLauncher(layer, impl, std::make_shared<OutMemoryInitializer>()) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:      : ReshapeLauncher(layer, impl, std::make_shared<OutMemoryInitializer>()) {" << std::endl;}

void OutMemoryReshapeLauncher::applyChanges(CNNLayer* layer) {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutMemoryReshapeLauncher::applyChanges(CNNLayer* layer) {" << std::endl;
    checkLayer(layer);
    _iController->applyChanges();
}

void OutMemoryReshapeLauncher::reset() {
    std::cerr << "./inference-engine/src/inference_engine/shape_infer/ie_reshape_launcher.cpp:  void OutMemoryReshapeLauncher::reset() {" << std::endl;
    _iController->reset();
}