#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_proposal_layer.hpp>
#include <ie_cnn_layer_builder.h>

#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::ProposalLayer::ProposalLayer(const std::string& name): LayerDecorator("Proposal", name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer::ProposalLayer(const std::string& name): LayerDecorator('Proposal', name) {" << std::endl;
    getLayer()->getOutputPorts().resize(1);
    getLayer()->getInputPorts().resize(3);
}

Builder::ProposalLayer::ProposalLayer(const Layer::Ptr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer::ProposalLayer(const Layer::Ptr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Proposal");
}

Builder::ProposalLayer::ProposalLayer(const Layer::CPtr& layer): LayerDecorator(layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer::ProposalLayer(const Layer::CPtr& layer): LayerDecorator(layer) {" << std::endl;
    checkType("Proposal");
}

Builder::ProposalLayer& Builder::ProposalLayer::setName(const std::string& name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setName(const std::string& name) {" << std::endl;
    getLayer()->setName(name);
    return *this;
}

const std::vector<Port>& Builder::ProposalLayer::getInputPorts() const {
    return getLayer()->getInputPorts();
}

Builder::ProposalLayer& Builder::ProposalLayer::setInputPorts(const std::vector<Port> &ports) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setInputPorts(const std::vector<Port> &ports) {" << std::endl;
    if (ports.size() != 3)
        THROW_IE_EXCEPTION << "Incorrect number of inputs for Proposal getLayer().";
    getLayer()->getInputPorts() = ports;
    return *this;
}

const Port& Builder::ProposalLayer::getOutputPort() const {
    return getLayer()->getOutputPorts()[0];
}

Builder::ProposalLayer& Builder::ProposalLayer::setOutputPort(const Port &port) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setOutputPort(const Port &port) {" << std::endl;
    getLayer()->getOutputPorts()[0] = port;
    return *this;
}

size_t Builder::ProposalLayer::getPostNMSTopN() const {
    return getLayer()->getParameters().at("post_nms_topn");
}
Builder::ProposalLayer& Builder::ProposalLayer::setPostNMSTopN(size_t topN) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setPostNMSTopN(size_t topN) {" << std::endl;
    getLayer()->getParameters()["post_nms_topn"] = topN;
    return *this;
}
size_t Builder::ProposalLayer::getPreNMSTopN() const {
    return getLayer()->getParameters().at("pre_nms_topn");
}
Builder::ProposalLayer& Builder::ProposalLayer::setPreNMSTopN(size_t topN) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setPreNMSTopN(size_t topN) {" << std::endl;
    getLayer()->getParameters()["pre_nms_topn"] = topN;
    return *this;
}
float Builder::ProposalLayer::getNMSThresh() const {
    return getLayer()->getParameters().at("nms_thresh");
}
Builder::ProposalLayer& Builder::ProposalLayer::setNMSThresh(float thresh) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setNMSThresh(float thresh) {" << std::endl;
    getLayer()->getParameters()["nms_thresh"] = thresh;
    return *this;
}
size_t Builder::ProposalLayer::getBaseSize() const {
    return getLayer()->getParameters().at("base_size");
}
Builder::ProposalLayer& Builder::ProposalLayer::setBaseSize(size_t baseSize) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setBaseSize(size_t baseSize) {" << std::endl;
    getLayer()->getParameters()["base_size"] = baseSize;
    return *this;
}
size_t Builder::ProposalLayer::getMinSize() const {
    return getLayer()->getParameters().at("min_size");
}
Builder::ProposalLayer& Builder::ProposalLayer::setMinSize(size_t minSize) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setMinSize(size_t minSize) {" << std::endl;
    getLayer()->getParameters()["min_size"] = minSize;
    return *this;
}
size_t Builder::ProposalLayer::getFeatStride() const {
    return getLayer()->getParameters().at("feat_stride");
}
Builder::ProposalLayer& Builder::ProposalLayer::setFeatStride(size_t featStride) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setFeatStride(size_t featStride) {" << std::endl;
    getLayer()->getParameters()["feat_stride"] = featStride;
    return *this;
}
const std::vector<float> Builder::ProposalLayer::getScale() const {
    return getLayer()->getParameters().at("scale");
}
Builder::ProposalLayer& Builder::ProposalLayer::setScale(const std::vector<float>& scales) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setScale(const std::vector<float>& scales) {" << std::endl;
    getLayer()->getParameters()["scale"] = scales;
    return *this;
}
const std::vector<float> Builder::ProposalLayer::getRatio() const {
    return getLayer()->getParameters().at("ratio");
}
Builder::ProposalLayer& Builder::ProposalLayer::setRatio(const std::vector<float>& ratios) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  Builder::ProposalLayer& Builder::ProposalLayer::setRatio(const std::vector<float>& ratios) {" << std::endl;
    getLayer()->getParameters()["ratio"] = ratios;
    return *this;
}

REG_CONVERTER_FOR(Proposal, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_proposal_layer.cpp:  REG_CONVERTER_FOR(Proposal, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    layer.getParameters()["post_nms_topn"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("post_nms_topn", 0));
    layer.getParameters()["pre_nms_topn"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("pre_nms_topn", 0));
    layer.getParameters()["nms_thresh"] = cnnLayer->GetParamAsFloat("nms_thresh", 0);
    layer.getParameters()["min_size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("base_size", 0));
    layer.getParameters()["max_size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("max_size", 0));
    layer.getParameters()["max_size"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("feat_stride", 0));
    layer.getParameters()["scale"] = cnnLayer->GetParamAsFloats("scale");
    layer.getParameters()["ratio"] = cnnLayer->GetParamAsFloats("ratio");
});