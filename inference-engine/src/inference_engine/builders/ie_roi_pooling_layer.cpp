#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_roi_pooling_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ROIPoolingLayer::ROIPoolingLayer(const std::string& name): LayerDecorator("ROIPooling", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp:  Builder::ROIPoolingLayer::ROIPoolingLayer(const std::string& name): LayerDecorator('ROIPooling', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    setPooled({0, 0});
}

Builder::ROIPoolingLayer::ROIPoolingLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp:  Builder::ROIPoolingLayer::ROIPoolingLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ROIPooling");
}

Builder::ROIPoolingLayer::ROIPoolingLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp:  Builder::ROIPoolingLayer::ROIPoolingLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("ROIPooling");
}

Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp:  Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}
const std::vector<Port>& Builder::ROIPoolingLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}
Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setInputPorts(const std::vector<Port>& ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp:  Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setInputPorts(const std::vector<Port>& ports) {" << std::endl;
    if (ports.size() != 2)
        THROW_IE_EXCEPTION << "ROIPoolingLayer should have 2 inputs!";
    getLayer()->getInputPorts() = ports;
    return *this;
}
const Port& Builder::ROIPoolingLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}
Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setOutputPort(const Port& port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp:  Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setOutputPort(const Port& port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}
float Builder::ROIPoolingLayer::getSpatialScale() const {
    return getLayer()->getParameters().at("spatial_scale");
}
Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setSpatialScale(float spatialScale) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp:  Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setSpatialScale(float spatialScale) {" << std::endl;
    getLayer()->getParameters()["spatial_scale"] = spatialScale;
    return *this;
}
const std::vector<int> Builder::ROIPoolingLayer::getPooled() const {
    return {getLayer()->getParameters().at("pooled_h"),
            getLayer()->getParameters().at("pooled_w")};
}
Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setPooled(const std::vector<int>& pooled) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp:  Builder::ROIPoolingLayer& Builder::ROIPoolingLayer::setPooled(const std::vector<int>& pooled) {" << std::endl;
    if (pooled.size() != 2)
        THROW_IE_EXCEPTION << "ROIPoolingLayer supports only pooled for height and width dimensions";
    getLayer()->getParameters()["pooled_h"] = pooled[0];
    getLayer()->getParameters()["pooled_w"] = pooled[1];
    return *this;
}

REG_CONVERTER_FOR(ROIPooling, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_roi_pooling_layer.cpp:  REG_CONVERTER_FOR(ROIPooling, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["pooled_h"] = cnnLayer->GetParamAsInt("pooled_h", 0);
    layer.getParameters()["pooled_w"] = cnnLayer->GetParamAsInt("pooled_w", 0);
    layer.getParameters()["spatial_scale"] = cnnLayer->GetParamAsFloat("spatial_scale");
});