#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_layer_decorator.hpp>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace details;

Builder::LayerDecorator::LayerDecorator(const std::string& type, const std::string& name):
        cLayer(nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_decorator.cpp:          cLayer(nullptr) {" << std::endl;
    layer = std::make_shared<Layer>(type, name);
}

Builder::LayerDecorator::LayerDecorator(const Layer::Ptr& layer): cLayer(nullptr), layer(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_decorator.cpp:  Builder::LayerDecorator::LayerDecorator(const Layer::Ptr& layer): cLayer(nullptr), layer(layer) {" << std::endl;}
Builder::LayerDecorator::LayerDecorator(const Layer::CPtr& layer): cLayer(layer), layer(nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_decorator.cpp:  Builder::LayerDecorator::LayerDecorator(const Layer::CPtr& layer): cLayer(layer), layer(nullptr) {" << std::endl;}

Builder::LayerDecorator::LayerDecorator(const Builder::LayerDecorator& rval) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_decorator.cpp:  Builder::LayerDecorator::LayerDecorator(const Builder::LayerDecorator& rval) {" << std::endl;
    *this = rval;
}

Builder::LayerDecorator& Builder::LayerDecorator::operator=(const Builder::LayerDecorator& rval) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_decorator.cpp:  Builder::LayerDecorator& Builder::LayerDecorator::operator=(const Builder::LayerDecorator& rval) {" << std::endl;
    layer = rval.layer;
    cLayer = rval.cLayer;
    return *this;
}

Builder::LayerDecorator::operator Builder::Layer() const {
    getLayer()->validate(true);
    return *getLayer();
}

Builder::LayerDecorator::operator Builder::Layer::Ptr() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_decorator.cpp:  Builder::LayerDecorator::operator Builder::Layer::Ptr() {" << std::endl;
    getLayer()->validate(true);
    return getLayer();
}

Builder::LayerDecorator::operator Builder::Layer::CPtr() const {
    getLayer()->validate(true);
    return getLayer();
}

const std::string& Builder::LayerDecorator::getType() const {
    return getLayer()->getType();
}
const std::string& Builder::LayerDecorator::getName() const {
    return getLayer()->getName();
}

Builder::Layer::Ptr& Builder::LayerDecorator::getLayer() {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_decorator.cpp:  Builder::Layer::Ptr& Builder::LayerDecorator::getLayer() {" << std::endl;
    if (!layer) THROW_IE_EXCEPTION << "Cannot get Layer::Ptr!";
    return layer;
}

const Builder::Layer::CPtr Builder::LayerDecorator::getLayer() const {
    if (!cLayer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_layer_decorator.cpp:      if (!cLayer) {" << std::endl;
        if (!layer) THROW_IE_EXCEPTION << "Cannot get Layer::CPtr!";
        return std::static_pointer_cast<const Layer>(layer);
    }
    return cLayer;
}

void Builder::LayerDecorator::checkType(const std::string& type) const {
    if (!details::CaselessEq<std::string>()(getLayer()->getType(), type))
        THROW_IE_EXCEPTION << "Cannot create " << type << " decorator for layer " << getLayer()->getType();
}
