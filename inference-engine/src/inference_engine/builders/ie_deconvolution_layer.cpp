#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_deconvolution_layer.hpp>
#include <ie_cnn_layer_builder.h>
#include <limits>
#include <vector>
#include <string>

using namespace InferenceEngine;

Builder::DeconvolutionLayer::DeconvolutionLayer(const std::string& name): ConvolutionLayer(name) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:  Builder::DeconvolutionLayer::DeconvolutionLayer(const std::string& name): ConvolutionLayer(name) {" << std::endl;
    getLayer()->setType("Deconvolution");
}
Builder::DeconvolutionLayer::DeconvolutionLayer(const Layer::Ptr& layer): ConvolutionLayer(layer->getName()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:  Builder::DeconvolutionLayer::DeconvolutionLayer(const Layer::Ptr& layer): ConvolutionLayer(layer->getName()) {" << std::endl;
    this->getLayer() = layer;
    checkType("Deconvolution");
}
Builder::DeconvolutionLayer::DeconvolutionLayer(const Layer::CPtr& layer): ConvolutionLayer(layer->getName()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:  Builder::DeconvolutionLayer::DeconvolutionLayer(const Layer::CPtr& layer): ConvolutionLayer(layer->getName()) {" << std::endl;
    this->getLayer().reset();
    cLayer = layer;
    checkType("Deconvolution");
}

REG_VALIDATOR_FOR(Deconvolution, [] (const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:  REG_VALIDATOR_FOR(Deconvolution, [] (const InferenceEngine::Builder::Layer::CPtr& layer, bool partial) {" << std::endl;
    // WA for old IRs
    if (layer->getParameters().find("kernel") == layer->getParameters().end() &&
        layer->getParameters().find("kernel-x") != layer->getParameters().end() &&
        layer->getParameters().find("kernel-y") != layer->getParameters().end())
        return;
    Builder::DeconvolutionLayer deconvBuilder(layer);
    std::vector<size_t> l_kernel = deconvBuilder.getKernel();
    std::vector<size_t> l_dilation = deconvBuilder.getDilation();
    std::vector<size_t> l_paddingBegin = deconvBuilder.getPaddingsBegin();
    std::vector<size_t> l_paddingEnd = deconvBuilder.getPaddingsEnd();
    std::vector<size_t> l_strides = deconvBuilder.getStrides();

    if (l_paddingBegin.empty() && !l_kernel.empty())
        l_paddingBegin.resize(l_kernel.size(), 0);
    if (l_paddingEnd.empty() && !l_kernel.empty())
        l_paddingEnd.resize(l_kernel.size(), 0);
    if (l_dilation.empty() && !l_kernel.empty())
        l_dilation.resize(l_kernel.size(), 1);
    if (l_strides.empty() && !l_kernel.empty())
        l_strides.resize(l_kernel.size(), 1);

    if (l_kernel.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      if (l_kernel.empty()) {" << std::endl;
        THROW_IE_EXCEPTION << "Kernel is empty!";
    }

    if (l_paddingBegin.size() != l_paddingEnd.size()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      if (l_paddingBegin.size() != l_paddingEnd.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Padding_begin dimension is not equal to padding_end dimension";
    }

    if (!l_paddingBegin.empty() && l_kernel.size() != l_paddingBegin.size()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      if (!l_paddingBegin.empty() && l_kernel.size() != l_paddingBegin.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Padding dimension is not equal to kernel dimension";
    }

    if (l_kernel.size() != l_strides.size()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      if (l_kernel.size() != l_strides.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Stride dimension is not equal to kernel dimension";
    }

    if (!l_dilation.empty() && l_kernel.size() != l_dilation.size()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      if (!l_dilation.empty() && l_kernel.size() != l_dilation.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Dilation dimension is not equal to kernel dimension";
    }

    if (deconvBuilder.getOutDepth() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      if (deconvBuilder.getOutDepth() == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "OutDepth parameter should be more than 0";
    }

    for (size_t kernel_dim : l_kernel) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      for (size_t kernel_dim : l_kernel) {" << std::endl;
        if (kernel_dim == 0) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:          if (kernel_dim == 0) {" << std::endl;
            THROW_IE_EXCEPTION << "Kernel dimensions should be more than 0";
        }
    }

    for (size_t i_stride : l_strides) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      for (size_t i_stride : l_strides) {" << std::endl;
        if (i_stride == 0) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:          if (i_stride == 0) {" << std::endl;
            THROW_IE_EXCEPTION << "Strides should be more than 0";
        }
    }

    for (size_t dil : l_dilation) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      for (size_t dil : l_dilation) {" << std::endl;
        if (dil == 0)
            THROW_IE_EXCEPTION << "Dilation should be more than 0";
    }

    if (!deconvBuilder.getGroup())
        THROW_IE_EXCEPTION << "Group should be more than 0";

    if (deconvBuilder.getInputPort().shape().empty())
        return;

    const size_t IC = deconvBuilder.getInputPort().shape()[1];
    if (IC % deconvBuilder.getGroup())
        THROW_IE_EXCEPTION << "Number of input channels (" << IC <<
                           ") is not divided by group number (" << deconvBuilder.getGroup() << ")";

    size_t weight_size = deconvBuilder.getOutDepth() * IC / deconvBuilder.getGroup();
    for (size_t kernel_dim : l_kernel) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      for (size_t kernel_dim : l_kernel) {" << std::endl;
        if (static_cast<double>(weight_size) * kernel_dim > std::numeric_limits<size_t>::max()) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:          if (static_cast<double>(weight_size) * kernel_dim > std::numeric_limits<size_t>::max()) {" << std::endl;
            THROW_IE_EXCEPTION << "Weight size exceeds the size_t max";
        }
        weight_size *= kernel_dim;
    }

    if (partial)
        return;

    const auto weights = layer->getInputPorts()[1].getData()->getData();
    if (weights->size() != weight_size) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      if (weights->size() != weight_size) {" << std::endl;
        THROW_IE_EXCEPTION << "Weight size is not correct!";
    }

    const auto biases = layer->getInputPorts()[2].getData()->getData();
    if (biases && biases->cbuffer() && biases->size() != deconvBuilder.getOutDepth())
        THROW_IE_EXCEPTION << "Biases size is incorrect!";
});

REG_CONVERTER_FOR(Deconvolution, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:  REG_CONVERTER_FOR(Deconvolution, [](const CNNLayerPtr& cnnLayer, Builder::Layer& layer) {" << std::endl;
    // WA for old IRs
    if (cnnLayer->params.find("kernel") == cnnLayer->params.end() &&
        cnnLayer->params.find("kernel-x") != cnnLayer->params.end() &&
        cnnLayer->params.find("kernel-y") != cnnLayer->params.end())
        return;
    std::vector<unsigned int> tmp = cnnLayer->GetParamAsUInts("kernel");
    std::vector<size_t> cur(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      for (size_t i = 0; i < tmp.size(); ++i) {" << std::endl;
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["kernel"] = cur;

    tmp = cnnLayer->GetParamAsUInts("strides");
    cur.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      for (size_t i = 0; i < tmp.size(); ++i) {" << std::endl;
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["strides"] = cur;

    tmp = cnnLayer->GetParamAsUInts("dilations");
    cur.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      for (size_t i = 0; i < tmp.size(); ++i) {" << std::endl;
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["dilations"] = cur;

    tmp = cnnLayer->GetParamAsUInts("pads_begin");
    cur.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      for (size_t i = 0; i < tmp.size(); ++i) {" << std::endl;
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["pads_begin"] = cur;

    tmp = cnnLayer->GetParamAsUInts("pads_end");
    cur.resize(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/builders/ie_deconvolution_layer.cpp:      for (size_t i = 0; i < tmp.size(); ++i) {" << std::endl;
        cur[i] = static_cast<size_t>(tmp[i]);
    }
    layer.getParameters()["pads_end"] = cur;

    layer.getParameters()["group"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("group"));
    layer.getParameters()["output"] = static_cast<size_t>(cnnLayer->GetParamAsUInt("output"));
});