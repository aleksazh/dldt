#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layers_internal.hpp"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "layer_transform.hpp"

namespace InferenceEngine {

template <class Layer>
int getKernel(const Layer& layer, size_t i) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:  int getKernel(const Layer& layer, size_t i) {" << std::endl;
    if (layer._dilation.size() > i && layer._dilation[i]) return (layer._kernel[i] - 1) * layer._dilation[i] + 1;
    return layer._kernel[i];
}

template <>
int getKernel(const PoolingLayer& layer, size_t i) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:  int getKernel(const PoolingLayer& layer, size_t i) {" << std::endl;
    return layer._kernel[i];
}

template <class Layer>
Paddings getPaddingsInternal(const Layer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:  Paddings getPaddingsInternal(const Layer& layer) {" << std::endl;
    std::string errorPrefix = "Failed to calculate padding for " + layer.type + ": ";
    try {
        const std::map<std::string, std::string>& params = layer.params;
        const std::vector<DataWeakPtr>& insData = layer.insData;
        auto it = params.find("auto_pad");
        if (it != params.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (it != params.end()) {" << std::endl;
            if (it->second == "valid") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:              if (it->second == 'valid') {" << std::endl;
                return {PropertyVector<unsigned>(layer._kernel.size(), 0u),
                        PropertyVector<unsigned>(layer._kernel.size(), 0u)};
            } else {
                if ((insData.size() > 3 || insData.empty()) && layer.type != "DeformableConvolution")
                    THROW_IE_EXCEPTION << "number of inputs should be in range [1, 3]";
                if ((insData.size() > 4 || insData.empty()) && layer.type == "DeformableConvolution")
                    THROW_IE_EXCEPTION << "number of inputs should be in range [2, 4]";
                auto firstInput = insData[0].lock();
                if (!firstInput) THROW_IE_EXCEPTION << "input is empty";
                auto shape = firstInput->getTensorDesc().getDims();
                auto shape_size = shape.size();
                if (shape_size < 4 || shape_size > 5) THROW_IE_EXCEPTION << "input shape must be 4D or 5D";

                std::vector<int> shapes;
                shapes.push_back(shape[shape_size - 1]);
                shapes.push_back(shape[shape_size - 2]);
                if (shape_size > 4) shapes.push_back(shape[shape_size - 3]);

                PropertyVector<unsigned int> pad_begin, pad_end;

                bool same_upper = it->second == "same_upper";
                bool same_lower = it->second == "same_lower";
                bool is_deconv = (layer.type == "Deconvolution");

                for (size_t i = 0; i < layer._kernel.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:                  for (size_t i = 0; i < layer._kernel.size(); i++) {" << std::endl;
                    float PA = 0;
                    int kernel = getKernel(layer, i);

                    int stride = layer._stride.size() > i ? layer._stride[i] : 1;
                    int sh = shapes[i];
                    if (is_deconv) sh *= stride;

                    int rm = sh % stride;
                    if (rm == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:                      if (rm == 0) {" << std::endl;
                        PA = std::max(kernel - stride, 0);
                    } else {
                        PA = std::max(kernel - rm, 0);
                    }
                    float p_begin = PA * 0.5f, p_end = PA - p_begin;

                    if (same_upper) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:                      if (same_upper) {" << std::endl;
                        p_begin = std::floor(p_begin);
                        p_end = std::ceil(p_end);
                    } else if (same_lower) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:                      } else if (same_lower) {" << std::endl;
                        p_begin = std::ceil(p_begin);
                        p_end = std::floor(p_end);
                    }
                    pad_begin.insert(i, static_cast<unsigned int>(p_begin));
                    pad_end.insert(i, static_cast<unsigned int>(p_end));
                }

                return {pad_begin, pad_end};
            }
        }
        return {layer._padding, layer._pads_end};
    } catch (const InferenceEngine::details::InferenceEngineException& iee) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:      } catch (const InferenceEngine::details::InferenceEngineException& iee) {" << std::endl;
        THROW_IE_EXCEPTION << errorPrefix << iee.what();
    }
}

class PaddingsUpdater {
    std::reference_wrapper<Paddings> pad;

public:
    explicit PaddingsUpdater(Paddings& pad): pad(pad) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:      explicit PaddingsUpdater(Paddings& pad): pad(pad) {" << std::endl;}
    template <class T>
    typename std::enable_if<!std::is_same<T, CNNLayer*>::value, bool>::type operator()(T& layer) const {
        pad.get() = getPaddingsInternal(*layer);
        return true;
    }
    bool operator()(CNNLayer* layer) const {
        THROW_IE_EXCEPTION << "padding calculation for layer: " << layer->name << "(" << layer->type << ") unsupported";
    }
};

Paddings getPaddingsImpl(const CNNLayer& layer) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:  Paddings getPaddingsImpl(const CNNLayer& layer) {" << std::endl;
    Paddings actual;
    details::visitActualLayer(std::tuple<DeformableConvolutionLayer*, DeconvolutionLayer*, ConvolutionLayer*,
                                         BinaryConvolutionLayer*, PoolingLayer*, CNNLayer*>(),
                              layer, PaddingsUpdater(actual));
    return actual;
}

int getNumIteration(const TensorIterator& tensorIterator) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:  int getNumIteration(const TensorIterator& tensorIterator) {" << std::endl;
    using PortMap = TensorIterator::PortMap;
    const auto isIterable = [](const PortMap& rule) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:      const auto isIterable = [](const PortMap& rule) {" << std::endl; return rule.axis != -1; };
    const auto getNumIterations = [&tensorIterator](const PortMap& rule, const DataPtr& iterableData) -> int {
        if (iterableData == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (iterableData == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << ": Iteration over an invalid data object (null pointer dereference)";
        }
        const auto& dimensions = iterableData->getDims();

        const auto axis = rule.axis;
        if (axis < 0 || static_cast<std::size_t>(axis) >= dimensions.size()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (axis < 0 || static_cast<std::size_t>(axis) >= dimensions.size()) {" << std::endl;
            THROW_IE_EXCEPTION << R"(: Invalid "axis" value in an iteration component: )"
                               << rule.axis  << ", dimensions number = " << dimensions.size() << " (out of range)";
        }
        const auto space = dimensions[axis];
        const int start = (rule.start < 0 ? (space + 1) : 0) + rule.start;
        const int end   = (rule.end   < 0 ? (space + 1) : 0) + rule.end;

        const auto stride = rule.stride;
        if (stride == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (stride == 0) {" << std::endl;
            THROW_IE_EXCEPTION << R"(: Invalid "stride" value in an iteration component: )" << rule.stride << " (infinite loop)";
        }
        const auto step = std::abs(stride);

        const auto src = stride < 0 ? end : start;
        const auto dst = stride < 0 ? start : end;
        const auto length = dst - src;
        if (src < 0 || src >= dst || dst > space || length < step) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (src < 0 || src >= dst || dst > space || length < step) {" << std::endl;
            THROW_IE_EXCEPTION << R"(: Invalid "start"/"stride"/"end" values in an iteration component)"
                               << ": \"start\" = " << rule.start << ", \"stride\" = " << rule.stride  << ", \"end\" = " << rule.end;
        }

        if (length % step != 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (length % step != 0) {" << std::endl;
            THROW_IE_EXCEPTION << ": Each iteration must be the same size: length (" << length << ") is not divisible by step (" << step << ")";
        }

        return static_cast<int>(length / step);
    };


    int numIterations = 1;
    bool isDefault = true;
    for (const auto& rule : tensorIterator.input_port_map) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:      for (const auto& rule : tensorIterator.input_port_map) {" << std::endl;
        if (!isIterable(rule)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (!isIterable(rule)) {" << std::endl;
            continue;
        }

        if (rule.from < 0 || rule.from >= tensorIterator.insData.size()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (rule.from < 0 || rule.from >= tensorIterator.insData.size()) {" << std::endl;
            THROW_IE_EXCEPTION << R"(: Invalid "from" value: "from" = )" << rule.from
                               << " inputs number = " << tensorIterator.insData.size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, tensorIterator.insData[rule.from].lock());
        if (isDefault) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (isDefault) {" << std::endl;
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          } else if (numIterations != currentNumIterations) {" << std::endl;
            THROW_IE_EXCEPTION << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    for (const auto& rule : tensorIterator.output_port_map) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:      for (const auto& rule : tensorIterator.output_port_map) {" << std::endl;
        if (!isIterable(rule)) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (!isIterable(rule)) {" << std::endl;
            continue;
        }

        if (rule.from < 0 || rule.from >= tensorIterator.outData.size()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (rule.from < 0 || rule.from >= tensorIterator.outData.size()) {" << std::endl;
            THROW_IE_EXCEPTION << R"(: Invalid "from" value: "from" = )" << rule.from
                               << " inputs number = " << tensorIterator.outData.size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, tensorIterator.outData[rule.from]);
        if (isDefault) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          if (isDefault) {" << std::endl;
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layers_internal.cpp:          } else if (numIterations != currentNumIterations) {" << std::endl;
            THROW_IE_EXCEPTION << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    return numIterations;
}


}  // namespace InferenceEngine
