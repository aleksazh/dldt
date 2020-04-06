#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_graph_dumper.h"
#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"
#include "exec_graph_info.hpp"
#include "mkldnn_debug.h"

#include <vector>
#include <string>
#include <memory>
#include <map>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

static void copy_node_metadata(const MKLDNNNodePtr &, CNNLayer::Ptr &);
static void drawer_callback(const InferenceEngine::CNNLayerPtr, ordered_properties &, ordered_properties &);

CNNLayer::Ptr convert_node(const MKLDNNNodePtr &node) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:  CNNLayer::Ptr convert_node(const MKLDNNNodePtr &node) {" << std::endl;
    CNNLayer::Ptr layer(new CNNLayer({"name", "type", Precision::FP32}));
    copy_node_metadata(node, layer);

    auto &cfg = node->getSelectedPrimitiveDescriptor()->getConfig();
    layer->insData.resize(cfg.inConfs.size());
    layer->outData.resize(cfg.outConfs.size());

    return layer;
}

std::shared_ptr<ICNNNetwork> dump_graph_as_ie_net(const MKLDNNGraph &graph) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:  std::shared_ptr<ICNNNetwork> dump_graph_as_ie_net(const MKLDNNGraph &graph) {" << std::endl;
    auto net = std::make_shared<details::CNNNetworkImpl>();

    net->setPrecision(Precision::FP32);
    net->setName(graph._name);
    std::map<MKLDNNNodePtr, CNNLayerPtr> node2layer;

    // Copy all nodes to network
    for (auto &node : graph.graphNodes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      for (auto &node : graph.graphNodes) {" << std::endl;
        auto layer = convert_node(node);
        node2layer[node] = layer;
        net->addLayer(layer);
    }

    // Copy all edges to network
    for (auto &node : graph.graphNodes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      for (auto &node : graph.graphNodes) {" << std::endl;
        auto pr = node2layer[node];
        auto ch_edges = node->getChildEdges();

        for (int i = 0; i < ch_edges.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:          for (int i = 0; i < ch_edges.size(); i++) {" << std::endl;
            auto edge = node->getChildEdgeAt(i);
            int out_port = edge->getInputNum();
            int in_port = edge->getOutputNum();
            auto ch_node = edge->getChild();
            auto ch  = node2layer[ch_node];

            DataPtr data;
            if (i < pr->outData.size()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:              if (i < pr->outData.size()) {" << std::endl;
                std::string data_name = node->getName() + "_out" + std::to_string(i);
                pr->outData[i] = std::make_shared<Data>(data_name, edge->getDesc());
                data = pr->outData[i];
                data->getCreatorLayer() = pr;
            } else {
                data = pr->outData[0];
            }

            data->getInputTo()[ch->name] = ch;
            ch->insData[in_port] = data;
        }
    }

    // Specify inputs data
    for (auto kvp : graph.inputNodes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      for (auto kvp : graph.inputNodes) {" << std::endl;
        auto in_node = kvp.second;
        auto in_layer = node2layer[in_node];

        auto in_info = std::make_shared<InputInfo>();
        in_info->setInputData(in_layer->outData[0]);
        net->setInputInfo(in_info);
    }

    return net;
}

void dump_graph_as_dot(const MKLDNNGraph &graph, std::ostream &out) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:  void dump_graph_as_dot(const MKLDNNGraph &graph, std::ostream &out) {" << std::endl;
    auto dump_net = dump_graph_as_ie_net(graph);
    InferenceEngine::saveGraphToDot(*dump_net, out, drawer_callback);
}

//**********************************
// Special converters of meta data
//**********************************

static const char BLUE[]  = "#D8D9F1";
static const char GREEN[] = "#D9EAD3";

void copy_node_metadata(const MKLDNNNodePtr &node, CNNLayer::Ptr &layer) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:  void copy_node_metadata(const MKLDNNNodePtr &node, CNNLayer::Ptr &layer) {" << std::endl;
    if (node->getType() == Input && node->isConstant()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      if (node->getType() == Input && node->isConstant()) {" << std::endl;
        // We need to separate Input and Const layers
        layer->type = "Const";
    } else if (node->getType() == Generic) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      } else if (node->getType() == Generic) {" << std::endl;
        // Path to print actual name for extension layers
        layer->type = node->getTypeStr();
    } else {
        layer->type = NameFromType(node->getType());
    }
    layer->name = node->getName();

    // Original layers
    layer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES] = node->getOriginalLayers();

    // Implementation type name
    layer->params[ExecGraphInfoSerialization::IMPL_TYPE] = node->getPrimitiveDescriptorType();

    std::string outputPrecisionsStr;
    if (!node->getChildEdges().empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      if (!node->getChildEdges().empty()) {" << std::endl;
        outputPrecisionsStr = node->getChildEdgeAt(0)->getDesc().getPrecision().name();

        bool isAllEqual = true;
        for (size_t i = 1; i < node->getChildEdges().size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:          for (size_t i = 1; i < node->getChildEdges().size(); i++) {" << std::endl;
            if (node->getChildEdgeAt(i-1)->getDesc().getPrecision() != node->getChildEdgeAt(i)->getDesc().getPrecision()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:              if (node->getChildEdgeAt(i-1)->getDesc().getPrecision() != node->getChildEdgeAt(i)->getDesc().getPrecision()) {" << std::endl;
                isAllEqual = false;
                break;
            }
        }

        // If all output precisions are the same, we store the name only once
        if (!isAllEqual) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:          if (!isAllEqual) {" << std::endl;
            for (size_t i = 1; i < node->getChildEdges().size(); i++)
                outputPrecisionsStr += "," + std::string(node->getChildEdgeAt(i)->getDesc().getPrecision().name());
        }
    } else {
        // Branch to correctly handle output nodes
        if (!node->getParentEdges().empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:          if (!node->getParentEdges().empty()) {" << std::endl;
            outputPrecisionsStr = node->getParentEdgeAt(0)->getDesc().getPrecision().name();
        }
    }
    layer->params[ExecGraphInfoSerialization::OUTPUT_PRECISIONS] = outputPrecisionsStr;

    std::string outputLayoutsStr;
    auto outLayouts = node->getSelectedPrimitiveDescriptor()->getOutputLayouts();
    if (!outLayouts.empty()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      if (!outLayouts.empty()) {" << std::endl;
        outputLayoutsStr = mkldnn_fmt2str(static_cast<mkldnn_memory_format_t>(outLayouts[0]));

        bool isAllEqual = true;
        for (size_t i = 1; i < outLayouts.size(); i++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:          for (size_t i = 1; i < outLayouts.size(); i++) {" << std::endl;
            if (outLayouts[i - 1] != outLayouts[i]) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:              if (outLayouts[i - 1] != outLayouts[i]) {" << std::endl;
                isAllEqual = false;
                break;
            }
        }

        // If all output layouts are the same, we store the name only once
        if (!isAllEqual) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:          if (!isAllEqual) {" << std::endl;
            for (size_t i = 1; i < outLayouts.size(); i++)
                outputLayoutsStr += "," + std::string(mkldnn_fmt2str(static_cast<mkldnn_memory_format_t>(outLayouts[i])));
        }
    } else {
        outputLayoutsStr = mkldnn_fmt2str(mkldnn_format_undef);
    }
    layer->params[ExecGraphInfoSerialization::OUTPUT_LAYOUTS] = outputLayoutsStr;

    // Performance
    if (node->PerfCounter().avg() != 0) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      if (node->PerfCounter().avg() != 0) {" << std::endl;
        layer->params[ExecGraphInfoSerialization::PERF_COUNTER] = std::to_string(node->PerfCounter().avg());
    } else {
        layer->params[ExecGraphInfoSerialization::PERF_COUNTER] = "not_executed";  // it means it was not calculated yet
    }

    layer->params[ExecGraphInfoSerialization::EXECUTION_ORDER] = std::to_string(node->getExecIndex());
}

void drawer_callback(const InferenceEngine::CNNLayerPtr layer,
        ordered_properties &printed_properties,
        ordered_properties &node_properties) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:          ordered_properties &node_properties) {" << std::endl;
    const auto &params = layer->params;

    // Implementation
    auto impl = params.find(ExecGraphInfoSerialization::IMPL_TYPE);
    if (impl != params.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      if (impl != params.end()) {" << std::endl;
        printed_properties.push_back({"impl", impl->second});
    }

    // Original names
    auto orig = params.find(ExecGraphInfoSerialization::ORIGINAL_NAMES);
    if (orig != params.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      if (orig != params.end()) {" << std::endl;
        printed_properties.push_back({"originals", orig->second});
    }

    // Precision
    auto prec = params.find(ExecGraphInfoSerialization::OUTPUT_PRECISIONS);
    if (prec != params.end()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_graph_dumper.cpp:      if (prec != params.end()) {" << std::endl;
        printed_properties.push_back({"precision", prec->second});
        // Set color
        node_properties.push_back({"fillcolor", prec->second == "FP32" ? GREEN : BLUE});
    }

    // Set xlabel containing PM data if calculated
    auto perf = layer->params.find(ExecGraphInfoSerialization::PERF_COUNTER);
    node_properties.push_back({"xlabel", (perf != layer->params.end()) ? perf->second : ""});
}

}  // namespace MKLDNNPlugin
