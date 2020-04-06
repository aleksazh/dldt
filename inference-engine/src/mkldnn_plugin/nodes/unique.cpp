#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <algorithm>
#include <functional>
#include <limits>
#include <utility>
#include "ie_parallel.hpp"
#include "common/simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class UniqueImpl : public ExtLayerBase {
public:
    explicit UniqueImpl(const CNNLayer* layer) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:      explicit UniqueImpl(const CNNLayer* layer) {" << std::endl;
        try {
            // check number of inputs and outputs
            if (layer->insData.size() != 1 || layer->outData.size() < 1 || layer->outData.size() > 3) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (layer->insData.size() != 1 || layer->outData.size() < 1 || layer->outData.size() > 3) {" << std::endl;
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";
            }

            // check precision of tensors
            Precision input_indices_precision = layer->insData[0].lock()->getTensorDesc().getPrecision();
            if (input_indices_precision != Precision::FP32) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (input_indices_precision != Precision::FP32) {" << std::endl;
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision. Only FP32 is supported!";
            }

            // check attributes
            sorted = layer->GetParamAsBool("sorted");
            return_inverse = layer->GetParamAsBool("return_inverse");
            return_counts = layer->GetParamAsBool("return_counts");

            // check that a real number of outputs matches one claimed by attributes
            size_t claimed_num_outputs = 1;
            if (return_inverse) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (return_inverse) {" << std::endl;
                claimed_num_outputs++;
            }
            if (return_counts) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (return_counts) {" << std::endl;
                claimed_num_outputs++;
            }
            if (layer->outData.size() != claimed_num_outputs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (layer->outData.size() != claimed_num_outputs) {" << std::endl;
                THROW_IE_EXCEPTION << layer->name << " A number of outputs claimed by attributes does not match a real number of outputs!";
            }

            // check dimensions of input tensors
            SizeVector input_dims = layer->insData[0].lock()->getTensorDesc().getDims();
            if (input_dims.size() != 1) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (input_dims.size() != 1) {" << std::endl;
                THROW_IE_EXCEPTION << layer->name << " Input must be 1-D tensor.";
            }
            num_elements = input_dims[0];

            // check dimensions of output tensors and its precisions
            size_t cur_output_port = 0;
            SizeVector output_uniques_dims = layer->outData[cur_output_port]->getTensorDesc().getDims();
            Precision output_uniques_precision = layer->outData[cur_output_port]->getTensorDesc().getPrecision();
            if (output_uniques_precision != Precision::FP32) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (output_uniques_precision != Precision::FP32) {" << std::endl;
                THROW_IE_EXCEPTION << layer->name << " Incorrect precision for output tensor of unique elements. Only FP32 is supported!";
            }
            if (output_uniques_dims.size() != 1 || output_uniques_dims[0] != num_elements) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (output_uniques_dims.size() != 1 || output_uniques_dims[0] != num_elements) {" << std::endl;
                THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for output tensor of unique elements.";
            }
            if (return_inverse) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (return_inverse) {" << std::endl;
                cur_output_port++;
                SizeVector output_indices_dims = layer->outData[cur_output_port]->getTensorDesc().getDims();
                Precision output_indices_precision = layer->outData[cur_output_port]->getTensorDesc().getPrecision();
                if (output_indices_precision != Precision::FP32) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:                  if (output_indices_precision != Precision::FP32) {" << std::endl;
                    THROW_IE_EXCEPTION << layer->name << " Incorrect precision for output tensor of indices. Only FP32 is supported!";
                }
                if (output_indices_dims.size() != 1 || output_indices_dims[0] != num_elements) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:                  if (output_indices_dims.size() != 1 || output_indices_dims[0] != num_elements) {" << std::endl;
                    THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for output tensor of indices.";
                }
            }
            if (return_counts) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (return_counts) {" << std::endl;
                cur_output_port++;
                SizeVector output_counts_dims = layer->outData[cur_output_port]->getTensorDesc().getDims();
                Precision output_counts_precision = layer->outData[cur_output_port]->getTensorDesc().getPrecision();
                if (output_counts_precision != Precision::FP32) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:                  if (output_counts_precision != Precision::FP32) {" << std::endl;
                    THROW_IE_EXCEPTION << layer->name << " Incorrect precision for output tensor of counts. Only FP32 is supported!";
                }
                if (output_counts_dims.size() != 1 || output_counts_dims[0] != num_elements) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:                  if (output_counts_dims.size() != 1 || output_counts_dims[0] != num_elements) {" << std::endl;
                    THROW_IE_EXCEPTION << layer->name << " Incorrect dimensions for output tensor of counts.";
                }
            }

            // add a layer configuration
            if (layer->outData.size() == 1) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (layer->outData.size() == 1) {" << std::endl;
                addConfig(layer,
                    { DataConfigurator(ConfLayout::PLN) },
                    { DataConfigurator(ConfLayout::PLN) });
            } else if (layer->outData.size() == 2) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              } else if (layer->outData.size() == 2) {" << std::endl;
                addConfig(layer,
                    { DataConfigurator(ConfLayout::PLN) },
                    { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) });
            } else if (layer->outData.size() == 3) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              } else if (layer->outData.size() == 3) {" << std::endl;
                addConfig(layer,
                    { DataConfigurator(ConfLayout::PLN) },
                    { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) });
            }
        }
        catch (InferenceEngine::details::InferenceEngineException &ex) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:          catch (InferenceEngine::details::InferenceEngineException &ex) {" << std::endl;
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const float *input_ptr = inputs[0]->cbuffer().as<const float *>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        size_t cur_output_port = 0;
        float *output_uniques_ptr = outputs[cur_output_port]->cbuffer().as<float *>() +
            outputs[cur_output_port]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float *output_indices_ptr = nullptr;
        if (return_inverse) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:          if (return_inverse) {" << std::endl;
            cur_output_port++;
            output_indices_ptr = outputs[cur_output_port]->cbuffer().as<float *>() +
                outputs[cur_output_port]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        }
        float *output_counts_ptr = nullptr;
        if (return_counts) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:          if (return_counts) {" << std::endl;
            cur_output_port++;
            output_counts_ptr = outputs[cur_output_port]->cbuffer().as<float *>() +
                outputs[cur_output_port]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        }

        // create a copy since input can be changed by sorting
        std::vector<float> input_copy(num_elements);
        std::copy(input_ptr, input_ptr + num_elements, input_copy.begin());

        // sort elements in the input copy
        if (sorted) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:          if (sorted) {" << std::endl;
            parallel_sort(input_copy.begin(), input_copy.end(), std::less<float>());
        }

        // walk through elements and save them along with its indice and occurences
        std::unordered_map<float, float> indices;
        for (size_t i = 0, num_unique_elements = 0; i < num_elements; i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:          for (size_t i = 0, num_unique_elements = 0; i < num_elements; i++) {" << std::endl;
            auto it = indices.find(input_copy[i]);
            if (it == indices.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              if (it == indices.end()) {" << std::endl;
                indices.insert(std::make_pair(input_copy[i], static_cast<float>(num_unique_elements)));
                output_uniques_ptr[num_unique_elements] = input_copy[i];
                if (return_inverse && !sorted) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:                  if (return_inverse && !sorted) {" << std::endl;
                    output_indices_ptr[i] = static_cast<float>(num_unique_elements);
                }
                if (return_counts) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:                  if (return_counts) {" << std::endl;
                    output_counts_ptr[num_unique_elements] = 1.0f;
                }
                num_unique_elements++;
            } else {
                if (return_inverse && !sorted) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:                  if (return_inverse && !sorted) {" << std::endl;
                    output_indices_ptr[i] = it->second;
                }
                if (return_counts) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:                  if (return_counts) {" << std::endl;
                    output_counts_ptr[static_cast<size_t>(it->second)] += 1.0f;
                }
            }
        }

        // compute indices individually when unique elements are known
        if (sorted && return_inverse) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:          if (sorted && return_inverse) {" << std::endl;
            for (size_t i = 0; i < num_elements; i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:              for (size_t i = 0; i < num_elements; i++) {" << std::endl;
                auto it = indices.find(input_ptr[i]);
                output_indices_ptr[i] = it->second;
            }
        }

        // fill a tail with the latest unique element used as an end mark
        size_t num_unique_elements = indices.size();
        if ((num_elements - num_unique_elements) > 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:          if ((num_elements - num_unique_elements) > 0) {" << std::endl;
            std::fill(output_uniques_ptr + num_unique_elements,
                output_uniques_ptr + num_elements,
                output_uniques_ptr[num_unique_elements - 1]);
        }

        // fill a tail for output buffer with counts
        if (return_counts && (num_elements - num_unique_elements) > 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/unique.cpp:          if (return_counts && (num_elements - num_unique_elements) > 0) {" << std::endl;
                std::fill(output_counts_ptr + num_unique_elements,
                    output_counts_ptr + num_elements, 0.f);
        }

        return OK;
    }

private:
    // attributes
    bool sorted;
    bool return_inverse;
    bool return_counts;

    size_t num_elements = 0;
};

REG_FACTORY_FOR(ImplFactory<UniqueImpl>, Unique);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
