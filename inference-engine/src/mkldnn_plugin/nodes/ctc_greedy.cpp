#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "list.hpp"
#include "base.hpp"

#include <cmath>
#include <vector>
#include <string>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class CTCGreedyDecoderImpl: public ExtLayerBase {
public:
    explicit CTCGreedyDecoderImpl(const CNNLayer* layer) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:      explicit CTCGreedyDecoderImpl(const CNNLayer* layer) {" << std::endl;
        try {
            if (layer->insData.empty() || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            std::vector<DataConfigurator> inps;
            inps.resize(layer->insData.size(), DataConfigurator(ConfLayout::PLN));
            addConfig(layer, inps, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:          } catch (InferenceEngine::details::InferenceEngineException &ex) {" << std::endl;
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        if ((inputs.size() != 1 && inputs.size() != 2) || outputs.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:          if ((inputs.size() != 1 && inputs.size() != 2) || outputs.empty()) {" << std::endl;
            if (resp) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:              if (resp) {" << std::endl;
                std::string errorMsg = "Incorrect number of input or output edges!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        const float* probabilities = inputs[0]->buffer();
        const float* sequence_indicators = inputs[1]->buffer();
        float* output_sequences = outputs[0]->buffer();

        size_t T_ = inputs[0]->getTensorDesc().getDims()[0];
        size_t N_ = inputs[0]->getTensorDesc().getDims()[1];
        size_t C_ = inputs[0]->getTensorDesc().getDims()[2];

        // Fill output_sequences with -1
        for (size_t ii = 0; ii < T_*N_; ii++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:          for (size_t ii = 0; ii < T_*N_; ii++) {" << std::endl;
            output_sequences[ii] = -1;
        }

        for (size_t n = 0; n < N_; ++n) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:          for (size_t n = 0; n < N_; ++n) {" << std::endl;
            int prev_class_idx = -1;
            size_t output_index = n*T_;

            for (int t = 0; /* check at end */; ++t) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:              for (int t = 0; /* check at end */; ++t) {" << std::endl;
                // get maximum probability and its index
                int max_class_idx = 0;

                const float* probs = probabilities + t*C_*N_ + n*C_;
                float max_prob = probs[0];
                ++probs;

                for (size_t c = 1; c < C_; ++c, ++probs) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:                  for (size_t c = 1; c < C_; ++c, ++probs) {" << std::endl;
                    if (*probs > max_prob) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:                      if (*probs > max_prob) {" << std::endl;
                        max_class_idx = static_cast<int>(c);
                        max_prob = *probs;
                    }
                }

                if (max_class_idx < static_cast<int>(C_) - 1 &&
                        max_class_idx != prev_class_idx) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:                          max_class_idx != prev_class_idx) {" << std::endl;
                    output_sequences[output_index] = static_cast<float>(max_class_idx);
                    output_index++;
                }

                prev_class_idx = max_class_idx;

                if (t + 1 == static_cast<int>(T_) || sequence_indicators[(t + 1)*N_ + n] == 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/nodes/ctc_greedy.cpp:                  if (t + 1 == static_cast<int>(T_) || sequence_indicators[(t + 1)*N_ + n] == 0) {" << std::endl;
                    break;
                }
            }
        }
        return OK;
    }
};

REG_FACTORY_FOR(ImplFactory<CTCGreedyDecoderImpl>, CTCGreedyDecoder);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
