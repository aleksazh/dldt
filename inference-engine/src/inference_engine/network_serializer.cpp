#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "network_serializer.h"

#include <fstream>
#include <map>
#include <queue>
#include <deque>
#include <string>
#include <vector>
#include <unordered_set>

#include "details/caseless.hpp"
#include "details/ie_cnn_network_tools.h"
#include "exec_graph_info.hpp"
#include "xml_parse_utils.h"

using namespace InferenceEngine;
using namespace details;

namespace {
template <typename T>
std::string arrayToIRProperty(const T& property) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:  std::string arrayToIRProperty(const T& property) {" << std::endl;
    std::string sProperty;
    for (size_t i = 0; i < property.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (size_t i = 0; i < property.size(); i++) {" << std::endl;
        sProperty = sProperty + std::to_string(property[i]) + std::string((i != property.size() - 1) ? "," : "");
    }
    return sProperty;
}

template <typename T>
std::string arrayRevertToIRProperty(const T& property) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:  std::string arrayRevertToIRProperty(const T& property) {" << std::endl;
    std::string sProperty;
    for (size_t i = 0; i < property.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (size_t i = 0; i < property.size(); i++) {" << std::endl;
        sProperty = sProperty + std::to_string(property[property.size() - i - 1]) +
                    std::string((i != property.size() - 1) ? "," : "");
    }
    return sProperty;
}

std::size_t updatePreProcInfo(const InferenceEngine::ICNNNetwork& network, pugi::xml_node& netXml,
                              const std::size_t weightsDataOffset) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                                const std::size_t weightsDataOffset) {" << std::endl;
    InputsDataMap inputInfo;
    network.getInputsInfo(inputInfo);

    // Assume that you preprocess only one input
    auto dataOffset = weightsDataOffset;
    for (auto ii : inputInfo) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (auto ii : inputInfo) {" << std::endl;
        const PreProcessInfo& pp = ii.second->getPreProcess();
        size_t nInChannels = pp.getNumberOfChannels();
        if (nInChannels) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (nInChannels) {" << std::endl;
            pugi::xml_node preproc = netXml.append_child("pre-process");

            preproc.append_attribute("reference-layer-name").set_value(ii.first.c_str());
            preproc.append_attribute("mean-precision").set_value(Precision(Precision::FP32).name());

            for (size_t ch = 0; ch < nInChannels; ch++) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (size_t ch = 0; ch < nInChannels; ch++) {" << std::endl;
                const PreProcessChannel::Ptr& preProcessChannel = pp[ch];
                auto channel = preproc.append_child("channel");
                channel.append_attribute("id").set_value(ch);

                auto mean = channel.append_child("mean");

                if (!preProcessChannel->meanData) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  if (!preProcessChannel->meanData) {" << std::endl;
                    mean.append_attribute("value").set_value(preProcessChannel->meanValue);
                } else {
                    auto size = preProcessChannel->meanData->byteSize();
                    mean.append_attribute("size").set_value(size);
                    mean.append_attribute("offset").set_value(dataOffset);
                    dataOffset += size;
                }

                if (1.f != preProcessChannel->stdScale) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  if (1.f != preProcessChannel->stdScale) {" << std::endl;
                    channel.append_child("scale").append_attribute("value").set_value(
                        std::to_string(preProcessChannel->stdScale).c_str());
                }
            }
        }
    }
    return dataOffset;
}

void updateStatisticsInfo(const InferenceEngine::ICNNNetwork& network, pugi::xml_node& netXml) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:  void updateStatisticsInfo(const InferenceEngine::ICNNNetwork& network, pugi::xml_node& netXml) {" << std::endl;
    // If statistics exists, add it to the file
    ICNNNetworkStats* netNodesStats = nullptr;
    auto stats = netXml.append_child("statistics");
    auto resultCode = network.getStats(&netNodesStats, nullptr);
    if (resultCode != StatusCode::OK) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      if (resultCode != StatusCode::OK) {" << std::endl;
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << resultCode
                           << "Can't get statistics info for serialization of the model";
    }
    const NetworkStatsMap statsmap = netNodesStats->getNodesStats();

    auto joinCommas = [&](const std::vector<float>& v) -> std::string {
        std::string res;

        for (size_t i = 0; i < v.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          for (size_t i = 0; i < v.size(); ++i) {" << std::endl;
            res += std::to_string(v[i]);
            if (i < v.size() - 1) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              if (i < v.size() - 1) {" << std::endl;
                res += ", ";
            }
        }

        return res;
    };

    for (const auto& itStats : statsmap) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (const auto& itStats : statsmap) {" << std::endl;
        auto layer = stats.append_child("layer");

        layer.append_child("name").text().set(itStats.first.c_str());

        layer.append_child("min").text().set(joinCommas(itStats.second->_minOutputs).c_str());
        layer.append_child("max").text().set(joinCommas(itStats.second->_maxOutputs).c_str());
    }
}

}  //  namespace

std::vector<CNNLayerPtr> NetworkSerializer::CNNNetSortTopologically(const ICNNNetwork& network) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:  std::vector<CNNLayerPtr> NetworkSerializer::CNNNetSortTopologically(const ICNNNetwork& network) {" << std::endl;
    std::vector<CNNLayerPtr> ordered;
    std::unordered_set<std::string> used;

    OutputsDataMap outputs;
    network.getOutputsInfo(outputs);

    InputsDataMap inputs;
    network.getInputsInfo(inputs);

    auto get_consumers = [](const CNNLayerPtr& node) -> std::vector<CNNLayerPtr> {
        std::vector<CNNLayerPtr> consumers;
        for (const auto & output : node->outData) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          for (const auto & output : node->outData) {" << std::endl;
            for (const auto &consumer : output->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (const auto &consumer : output->getInputTo()) {" << std::endl;
                consumers.push_back(consumer.second);
            }
        }
        return consumers;
    };
    auto bfs = [&used, &ordered, &get_consumers](const CNNLayerPtr& start_node, bool traverse_via_outputs = false) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      auto bfs = [&used, &ordered, &get_consumers](const CNNLayerPtr& start_node, bool traverse_via_outputs = false) {" << std::endl;
        if (!start_node) return;
        std::deque<CNNLayerPtr> q;
        q.push_front(start_node);
        used.insert(start_node->name);
        while (!q.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          while (!q.empty()) {" << std::endl;
            auto node = q.front();
            q.pop_front();
            ordered.push_back(node);

            // Traverse via inputs
            for (const auto & input : node->insData) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (const auto & input : node->insData) {" << std::endl;
                auto locked_input = input.lock();
                if (!locked_input) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  if (!locked_input) {" << std::endl;
                    THROW_IE_EXCEPTION << "insData for " << node->name << " is not valid.";
                }
                if (auto next_node = locked_input->getCreatorLayer().lock()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  if (auto next_node = locked_input->getCreatorLayer().lock()) {" << std::endl;
                    if (!used.count(next_node->name)) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                      if (!used.count(next_node->name)) {" << std::endl;
                        // Check that all consumers were used
                        bool all_consumers_used(true);
                        for (const auto & consumer : get_consumers(next_node)) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                          for (const auto & consumer : get_consumers(next_node)) {" << std::endl;
                            if (!used.count(consumer->name)) all_consumers_used = false;
                        }
                        if (all_consumers_used) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                          if (all_consumers_used) {" << std::endl;
                            q.push_front(next_node);
                            used.insert(next_node->name);
                        }
                    }
                }
            }

            // Traverse via outputs
            if (traverse_via_outputs) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              if (traverse_via_outputs) {" << std::endl;
                for (const auto &consumer : get_consumers(node)) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  for (const auto &consumer : get_consumers(node)) {" << std::endl;
                    if (!used.count(consumer->name)) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                      if (!used.count(consumer->name)) {" << std::endl;
                        q.push_front(consumer);
                        used.insert(consumer->name);
                    }
                }
            }
        }
    };

    // First we run bfs starting from outputs that provides deterministic graph traverse
    for (const auto & output : outputs) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (const auto & output : outputs) {" << std::endl;
        if (!used.count(output.first)) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!used.count(output.first)) {" << std::endl;
            bfs(output.second->getCreatorLayer().lock());
        }
    }

    // For cases when graph has no outputs we start bfs from inputs to ensure topological sort
    for (const auto & input : inputs) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (const auto & input : inputs) {" << std::endl;
        const auto data_ptr = input.second->getInputData();
        for (const auto & consumer : data_ptr->getInputTo())
        if (!used.count(consumer.first)) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!used.count(consumer.first)) {" << std::endl;
            bfs(consumer.second, true);
        }
    }

    std::reverse(ordered.begin(), ordered.end());
    return ordered;
}


std::size_t NetworkSerializer::fillXmlDoc(const InferenceEngine::ICNNNetwork& network, pugi::xml_document& doc,
                                          const bool execGraphInfoSerialization, const bool dumpWeights) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                                            const bool execGraphInfoSerialization, const bool dumpWeights) {" << std::endl;
    const std::vector<CNNLayerPtr> ordered = NetworkSerializer::CNNNetSortTopologically(network);
    pugi::xml_node netXml = doc.append_child("net");
    netXml.append_attribute("name").set_value(network.getName().c_str());

    // no need to print this information for executable graph information serialization because it is not IR.
    if (!execGraphInfoSerialization) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      if (!execGraphInfoSerialization) {" << std::endl;
        netXml.append_attribute("version").set_value("6");
        netXml.append_attribute("batch").set_value(network.getBatchSize());
    }

    pugi::xml_node layers = netXml.append_child("layers");

    std::map<CNNLayer::Ptr, size_t> matching;
    for (size_t i = 0; i < ordered.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (size_t i = 0; i < ordered.size(); i++) {" << std::endl;
        matching[ordered[i]] = i;
    }

    const std::string dataName = "data";
    size_t dataOffset = 0;
    for (size_t i = 0; i < ordered.size(); ++i) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (size_t i = 0; i < ordered.size(); ++i) {" << std::endl;
        const CNNLayerPtr node = ordered[i];

        pugi::xml_node layer = layers.append_child("layer");
        const Precision precision = node->precision;
        layer.append_attribute("name").set_value(node->name.c_str());
        layer.append_attribute("type").set_value(node->type.c_str());
        layer.append_attribute("precision").set_value(precision.name());
        layer.append_attribute("id").set_value(i);

        if (!execGraphInfoSerialization) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!execGraphInfoSerialization) {" << std::endl;
            NetworkSerializer::updateStdLayerParams(node);
        }

        const auto& params = node->params;
        if (!params.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!params.empty()) {" << std::endl;
            pugi::xml_node data = layer.append_child(dataName.c_str());

            for (const auto& it : params) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (const auto& it : params) {" << std::endl;
                data.append_attribute(it.first.c_str()).set_value(it.second.c_str());
            }
        }

        if (!node->insData.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!node->insData.empty()) {" << std::endl;
            pugi::xml_node input = layer.append_child("input");

            for (size_t iport = 0; iport < node->insData.size(); iport++) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (size_t iport = 0; iport < node->insData.size(); iport++) {" << std::endl;
                const DataPtr d = node->insData[iport].lock();
                pugi::xml_node port = input.append_child("port");

                port.append_attribute("id").set_value(iport);

                for (auto dim : d->getDims()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  for (auto dim : d->getDims()) {" << std::endl;
                    port.append_child("dim").text().set(dim);
                }
            }
        }
        if (!node->outData.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!node->outData.empty()) {" << std::endl;
            pugi::xml_node output = layer.append_child("output");
            for (size_t oport = 0; oport < node->outData.size(); oport++) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (size_t oport = 0; oport < node->outData.size(); oport++) {" << std::endl;
                pugi::xml_node port = output.append_child("port");

                port.append_attribute("id").set_value(node->insData.size() + oport);
                port.append_attribute("precision").set_value(node->outData[oport]->getPrecision().name());

                for (const auto dim : node->outData[oport]->getDims()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  for (const auto dim : node->outData[oport]->getDims()) {" << std::endl;
                    port.append_child("dim").text().set(dim);
                }
            }
        }
        if (dumpWeights && !node->blobs.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (dumpWeights && !node->blobs.empty()) {" << std::endl;
            auto blobsNode = layer.append_child("blobs");
            for (const auto& dataIt : node->blobs) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (const auto& dataIt : node->blobs) {" << std::endl;
                size_t dataSize = dataIt.second->byteSize();
                pugi::xml_node data = blobsNode.append_child(dataIt.first.c_str());
                data.append_attribute("offset").set_value(dataOffset);
                data.append_attribute("size").set_value(dataSize);
                data.append_attribute("precision").set_value(dataIt.second->getTensorDesc().getPrecision().name());

                dataOffset += dataSize;
            }
        }
    }

    pugi::xml_node edges = netXml.append_child("edges");

    for (const auto& ord : ordered) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (const auto& ord : ordered) {" << std::endl;
        const CNNLayer::Ptr node = ord;

        if (!node->outData.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!node->outData.empty()) {" << std::endl;
            auto itFrom = matching.find(node);
            if (itFrom == matching.end()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              if (itFrom == matching.end()) {" << std::endl;
                THROW_IE_EXCEPTION << "Internal error, cannot find " << node->name
                                   << " in matching container during serialization of IR";
            }
            for (size_t oport = 0; oport < node->outData.size(); oport++) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (size_t oport = 0; oport < node->outData.size(); oport++) {" << std::endl;
                const DataPtr outData = node->outData[oport];
                for (const auto& inputTo : outData->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  for (const auto& inputTo : outData->getInputTo()) {" << std::endl;
                    for (int iport = 0; iport < inputTo.second->insData.size(); iport++) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                      for (int iport = 0; iport < inputTo.second->insData.size(); iport++) {" << std::endl;
                        if (inputTo.second->insData[iport].lock() == outData) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                          if (inputTo.second->insData[iport].lock() == outData) {" << std::endl;
                            auto itTo = matching.find(inputTo.second);
                            if (itTo == matching.end()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                              if (itTo == matching.end()) {" << std::endl;
                                THROW_IE_EXCEPTION << "Broken edge form layer " << node->name << " to layer "
                                                   << inputTo.first << "during serialization of IR";
                            }
                            pugi::xml_node edge = edges.append_child("edge");
                            edge.append_attribute("from-layer").set_value(itFrom->second);
                            edge.append_attribute("from-port").set_value(oport + node->insData.size());

                            edge.append_attribute("to-layer").set_value(itTo->second);
                            edge.append_attribute("to-port").set_value(iport);
                        }
                    }
                }
            }
        }
    }

    // no need to print this info in case of executable graph info serialization
    if (!execGraphInfoSerialization) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      if (!execGraphInfoSerialization) {" << std::endl;
        dataOffset = updatePreProcInfo(network, netXml, dataOffset);
        updateStatisticsInfo(network, netXml);
    }

    return dataOffset;
}

void NetworkSerializer::serializeBlobs(std::ostream& stream, const InferenceEngine::ICNNNetwork& network) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:  void NetworkSerializer::serializeBlobs(std::ostream& stream, const InferenceEngine::ICNNNetwork& network) {" << std::endl;
    const std::vector<CNNLayerPtr> ordered = NetworkSerializer::CNNNetSortTopologically(network);
    for (auto&& node : ordered) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (auto&& node : ordered) {" << std::endl;
        if (!node->blobs.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!node->blobs.empty()) {" << std::endl;
            for (const auto& dataIt : node->blobs) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (const auto& dataIt : node->blobs) {" << std::endl;
                const char* dataPtr = dataIt.second->buffer().as<char*>();
                size_t dataSize = dataIt.second->byteSize();
                stream.write(dataPtr, dataSize);
                if (!stream.good()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  if (!stream.good()) {" << std::endl;
                    THROW_IE_EXCEPTION << "Error during writing blob waights";
                }
            }
        }
    }

    InputsDataMap inputInfo;
    network.getInputsInfo(inputInfo);

    for (auto ii : inputInfo) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      for (auto ii : inputInfo) {" << std::endl;
        const PreProcessInfo& pp = ii.second->getPreProcess();
        size_t nInChannels = pp.getNumberOfChannels();
        if (nInChannels) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (nInChannels) {" << std::endl;
            for (size_t ch = 0; ch < nInChannels; ch++) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              for (size_t ch = 0; ch < nInChannels; ch++) {" << std::endl;
                const PreProcessChannel::Ptr& preProcessChannel = pp[ch];
                if (preProcessChannel->meanData) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                  if (preProcessChannel->meanData) {" << std::endl;
                    const char* dataPtr = preProcessChannel->meanData->buffer().as<char*>();
                    size_t dataSize = preProcessChannel->meanData->byteSize();
                    stream.write(dataPtr, dataSize);
                    if (!stream.good()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                      if (!stream.good()) {" << std::endl;
                        THROW_IE_EXCEPTION << "Error during writing mean data";
                    }
                }
            }
        }
    }
}

void NetworkSerializer::serialize(const std::string& xmlPath, const std::string& binPath,
                                  const InferenceEngine::ICNNNetwork& network) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                                    const InferenceEngine::ICNNNetwork& network) {" << std::endl;
    const std::vector<CNNLayerPtr> ordered = NetworkSerializer::CNNNetSortTopologically(network);

    // A flag for serializing executable graph information (not complete IR)
    bool execGraphInfoSerialization = false;
    // If first layer has perfCounter parameter set then it's executable graph info serialization.
    // All other layers must also have this parameter set.
    if (ordered[0]->params.find(ExecGraphInfoSerialization::PERF_COUNTER) != ordered[0]->params.end()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      if (ordered[0]->params.find(ExecGraphInfoSerialization::PERF_COUNTER) != ordered[0]->params.end()) {" << std::endl;
        execGraphInfoSerialization = true;
        for (const auto& layer : ordered) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          for (const auto& layer : ordered) {" << std::endl;
            if (layer->params.find(ExecGraphInfoSerialization::PERF_COUNTER) == layer->params.end()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:              if (layer->params.find(ExecGraphInfoSerialization::PERF_COUNTER) == layer->params.end()) {" << std::endl;
                THROW_IE_EXCEPTION << "Each node must have " << ExecGraphInfoSerialization::PERF_COUNTER
                                   << " parameter set in case of executable graph info serialization";
            }
        }
    }

    bool dumpWeights = !execGraphInfoSerialization & !binPath.empty();

    pugi::xml_document doc;
    fillXmlDoc(network, doc, execGraphInfoSerialization, dumpWeights);

    if (!doc.save_file(xmlPath.c_str())) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      if (!doc.save_file(xmlPath.c_str())) {" << std::endl;
        THROW_IE_EXCEPTION << "file '" << xmlPath << "' was not serialized";
    }

    std::ofstream ofsBin;
    if (dumpWeights) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      if (dumpWeights) {" << std::endl;
        ofsBin.open(binPath, std::ofstream::out | std::ofstream::binary);
        if (!ofsBin) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!ofsBin) {" << std::endl;
            THROW_IE_EXCEPTION << "File '" << binPath << "' is not opened as out file stream";
        }
        serializeBlobs(ofsBin, network);
        ofsBin.close();
        if (!ofsBin.good()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (!ofsBin.good()) {" << std::endl;
            THROW_IE_EXCEPTION << "Error during '" << binPath << "' closing";
        }
    }
}

void NetworkSerializer::updateStdLayerParams(const CNNLayer::Ptr& layer) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:  void NetworkSerializer::updateStdLayerParams(const CNNLayer::Ptr& layer) {" << std::endl;
    auto layerPtr = layer.get();
    auto& params = layer->params;

    if (CaselessEq<std::string>()(layer->type, "power")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      if (CaselessEq<std::string>()(layer->type, 'power')) {" << std::endl;
        auto* lr = dynamic_cast<PowerLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of PowerLayer class";
        }
        params["scale"] = std::to_string(lr->scale);
        params["shift"] = std::to_string(lr->offset);
        params["power"] = std::to_string(lr->power);
    } else if (CaselessEq<std::string>()(layer->type, "convolution") ||
               CaselessEq<std::string>()(layer->type, "deconvolution")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                 CaselessEq<std::string>()(layer->type, 'deconvolution')) {" << std::endl;
        auto* lr = dynamic_cast<ConvolutionLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ConvolutionLayer class";
        }
        params["kernel"] = arrayRevertToIRProperty(lr->_kernel);
        params["pads_begin"] = arrayRevertToIRProperty(lr->_padding);
        params["pads_end"] = arrayRevertToIRProperty(lr->_pads_end);
        params["strides"] = arrayRevertToIRProperty(lr->_stride);
        params["dilations"] = arrayRevertToIRProperty(lr->_dilation);
        params["output"] = std::to_string(lr->_out_depth);
        params["group"] = std::to_string(lr->_group);
    } else if (CaselessEq<std::string>()(layer->type, "deformable_convolution")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'deformable_convolution')) {" << std::endl;
        auto* lr = dynamic_cast<DeformableConvolutionLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of DeformableConvolutionLayer class";
        }
        params["kernel"] = arrayRevertToIRProperty(lr->_kernel);
        params["pads_begin"] = arrayRevertToIRProperty(lr->_padding);
        params["pads_end"] = arrayRevertToIRProperty(lr->_pads_end);
        params["strides"] = arrayRevertToIRProperty(lr->_stride);
        params["dilations"] = arrayRevertToIRProperty(lr->_dilation);
        params["output"] = std::to_string(lr->_out_depth);
        params["group"] = std::to_string(lr->_group);
        params["deformable_group"] = std::to_string(lr->_deformable_group);
    } else if (CaselessEq<std::string>()(layer->type, "relu")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'relu')) {" << std::endl;
        auto* lr = dynamic_cast<ReLULayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ReLULayer class";
        }
        if (lr->negative_slope != 0.0f) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr->negative_slope != 0.0f) {" << std::endl;
            params["negative_slope"] = std::to_string(lr->negative_slope);
        }
    } else if (CaselessEq<std::string>()(layer->type, "norm") || CaselessEq<std::string>()(layer->type, "lrn")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'norm') || CaselessEq<std::string>()(layer->type, 'lrn')) {" << std::endl;
        auto* lr = dynamic_cast<NormLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of NormLayer class";
        }
        params["alpha"] = std::to_string(lr->_alpha);
        params["beta"] = std::to_string(lr->_beta);
        params["local-size"] = std::to_string(lr->_size);
        params["region"] = lr->_isAcrossMaps ? "across" : "same";
    } else if (CaselessEq<std::string>()(layer->type, "pooling")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'pooling')) {" << std::endl;
        auto* lr = dynamic_cast<PoolingLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of PoolingLayer class";
        }
        params["kernel"] = arrayRevertToIRProperty(lr->_kernel);
        params["pads_begin"] = arrayRevertToIRProperty(lr->_padding);
        params["pads_end"] = arrayRevertToIRProperty(lr->_pads_end);
        params["strides"] = arrayRevertToIRProperty(lr->_stride);

        switch (lr->_type) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          switch (lr->_type) {" << std::endl;
        case PoolingLayer::MAX:
            params["pool-method"] = "max";
            break;
        case PoolingLayer::AVG:
            params["pool-method"] = "avg";
            break;

        default:
            THROW_IE_EXCEPTION << "Found unsupported pooling method: " << lr->_type;
        }
    } else if (CaselessEq<std::string>()(layer->type, "split")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'split')) {" << std::endl;
        auto* lr = dynamic_cast<SplitLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of SplitLayer class";
        }
        params["axis"] = std::to_string(lr->_axis);
    } else if (CaselessEq<std::string>()(layer->type, "concat")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'concat')) {" << std::endl;
        auto* lr = dynamic_cast<ConcatLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ConcatLayer class";
        }
        params["axis"] = std::to_string(lr->_axis);
    } else if (CaselessEq<std::string>()(layer->type, "FullyConnected") ||
               CaselessEq<std::string>()(layer->type, "InnerProduct")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                 CaselessEq<std::string>()(layer->type, 'InnerProduct')) {" << std::endl;
        auto* lr = dynamic_cast<FullyConnectedLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of FullyConnectedLayer class";
        }
        params["out-size"] = std::to_string(lr->_out_num);
    } else if (CaselessEq<std::string>()(layer->type, "softmax")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'softmax')) {" << std::endl;
        auto* lr = dynamic_cast<SoftMaxLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of SoftMaxLayer class";
        }
        params["axis"] = std::to_string(lr->axis);
    } else if (CaselessEq<std::string>()(layer->type, "reshape")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'reshape')) {" << std::endl;
        // need to add here support of flatten layer if it is created from API
        auto* lr = dynamic_cast<ReshapeLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ReshapeLayer class";
        }
        params["dim"] = arrayToIRProperty(lr->shape);
    } else if (CaselessEq<std::string>()(layer->type, "Eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'Eltwise')) {" << std::endl;
        auto* lr = dynamic_cast<EltwiseLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of EltwiseLayer class";
        }

        std::string op;

        switch (lr->_operation) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          switch (lr->_operation) {" << std::endl;
        case EltwiseLayer::Sum:
            op = "sum";
            break;
        case EltwiseLayer::Prod:
            op = "prod";
            break;
        case EltwiseLayer::Max:
            op = "max";
            break;
        case EltwiseLayer::Sub:
            op = "sub";
            break;
        default:
            break;
        }

        params["operation"] = op;
    } else if (CaselessEq<std::string>()(layer->type, "scaleshift")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'scaleshift')) {" << std::endl;
        auto* lr = dynamic_cast<ScaleShiftLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ScaleShiftLayer class";
        }
        params["broadcast"] = std::to_string(lr->_broadcast);
    } else if (CaselessEq<std::string>()(layer->type, "crop")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'crop')) {" << std::endl;
        auto* lr = dynamic_cast<CropLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of CropLayer class";
        }
        params["axis"] = arrayToIRProperty(lr->axis);
        params["offset"] = arrayToIRProperty(lr->offset);
        params["dim"] = arrayToIRProperty(lr->dim);
    } else if (CaselessEq<std::string>()(layer->type, "tile")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'tile')) {" << std::endl;
        auto* lr = dynamic_cast<TileLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of TileLayer class";
        }
        params["axis"] = std::to_string(lr->axis);
        params["tiles"] = std::to_string(lr->tiles);
    } else if (CaselessEq<std::string>()(layer->type, "prelu")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'prelu')) {" << std::endl;
        auto* lr = dynamic_cast<PReLULayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of PReLULayer class";
        }
        params["channel_shared"] = std::to_string(lr->_channel_shared);
    } else if (CaselessEq<std::string>()(layer->type, "clamp")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'clamp')) {" << std::endl;
        auto* lr = dynamic_cast<ClampLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ClampLayer class";
        }
        params["min"] = std::to_string(lr->min_value);
        params["max"] = std::to_string(lr->max_value);
    } else if (CaselessEq<std::string>()(layer->type, "BatchNormalization")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'BatchNormalization')) {" << std::endl;
        auto* lr = dynamic_cast<BatchNormalizationLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of BatchNormalizationLayer class";
        }
        params["epsilon"] = std::to_string(lr->epsilon);
    } else if (CaselessEq<std::string>()(layer->type, "grn")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'grn')) {" << std::endl;
        auto* lr = dynamic_cast<GRNLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of GRNLayer class";
        }
        params["bias"] = std::to_string(lr->bias);
    } else if (CaselessEq<std::string>()(layer->type, "mvn")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'mvn')) {" << std::endl;
        auto* lr = dynamic_cast<MVNLayer*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of MVNLayer class";
        }
        params["across_channels"] = std::to_string(lr->across_channels);
        params["normalize_variance"] = std::to_string(lr->normalize);
    } else if (CaselessEq<std::string>()(layer->type, "LSTMCell")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      } else if (CaselessEq<std::string>()(layer->type, 'LSTMCell')) {" << std::endl;
        auto* lr = dynamic_cast<RNNCellBase*>(layerPtr);
        if (lr == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (lr == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of LSTMCell class";
        }
        params["hidden_size"] = std::to_string(lr->hidden_size);
    } else if (CaselessEq<std::string>()(layer->type, "rnn") ||
               CaselessEq<std::string>()(layer->type, "TensorIterator")) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:                 CaselessEq<std::string>()(layer->type, 'TensorIterator')) {" << std::endl;
        THROW_IE_EXCEPTION << "Not covered layers for writing to IR";
    }

    if (layer->params.find("quantization_level") != layer->params.end()) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      if (layer->params.find('quantization_level') != layer->params.end()) {" << std::endl;
        params["quantization_level"] = layer->params["quantization_level"];
    }

    // update of weightable layers
    auto* pwlayer = dynamic_cast<WeightableLayer*>(layerPtr);
    if (pwlayer) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:      if (pwlayer) {" << std::endl;
        if (pwlayer->_weights) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (pwlayer->_weights) {" << std::endl;
            pwlayer->blobs["weights"] = pwlayer->_weights;
        }
        if (pwlayer->_biases) {
    std::cerr << "./inference-engine/src/inference_engine/network_serializer.cpp:          if (pwlayer->_biases) {" << std::endl;
            pwlayer->blobs["biases"] = pwlayer->_biases;
        }
    }
}
