#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layer_parsers.h"

#include <memory>
#include <set>
#include <string>
#include <utility>

#include "ie_cnn_net_reader_impl.h"

namespace InferenceEngine {
namespace details {

CNNLayer::Ptr ActivationLayerCreator::CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:  CNNLayer::Ptr ActivationLayerCreator::CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) {" << std::endl;
    pugi::xml_node dn = GetChild(node, {"data", "activation_data"}, false);
    if (dn.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      if (dn.empty()) {" << std::endl;
        THROW_IE_EXCEPTION << "Activation layer has no data node";
    }

    std::string type;
    for (auto ait = dn.attributes_begin(); ait != dn.attributes_end(); ++ait) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      for (auto ait = dn.attributes_begin(); ait != dn.attributes_end(); ++ait) {" << std::endl;
        pugi::xml_attribute attr = *ait;
        if (CaselessEq<std::string>()("type", attr.name())) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:          if (CaselessEq<std::string>()('type', attr.name())) {" << std::endl;
            if (!type.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:              if (!type.empty()) {" << std::endl;
                THROW_IE_EXCEPTION << "Activation layer has multiple types";
            }
            type = attr.value();
        }
    }

    static caseless_map<std::string, std::shared_ptr<BaseCreator>> activationCreators = {
        {"relu", std::make_shared<LayerCreator<ReLULayer>>("ReLU")},
        {"relu6", std::make_shared<LayerCreator<ReLU6Layer>>("ReLU6")},
        {"prelu", std::make_shared<LayerCreator<PReLULayer>>("PReLU")},
        {"clamp", std::make_shared<LayerCreator<ClampLayer>>("Clamp")},
        {"elu", std::make_shared<LayerCreator<CNNLayer>>("ELU")},
        {"sigmoid", std::make_shared<LayerCreator<CNNLayer>>("Sigmoid")},
        {"tanh", std::make_shared<LayerCreator<CNNLayer>>("TanH")},
    };

    CNNLayer::Ptr activation;

    auto activationBuilder = activationCreators.find(type);
    if (activationBuilder == activationCreators.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      if (activationBuilder == activationCreators.end()) {" << std::endl;
        auto activationCreator = std::make_shared<LayerCreator<CNNLayer>>(type);
        if (!activationCreator) THROW_IE_EXCEPTION << "Cannot create activation layer with type " << type;

        activation = activationCreator->CreateLayer(node, layerParsePrms);
        activation->type = type;
    } else {
        activation = activationBuilder->second->CreateLayer(node, layerParsePrms);
        activation->type = activationBuilder->first;
    }

    activation->params.erase("type");

    return activation;
}

/***********************************************************************************/
/*******  Tensor Iterator parser  **************************************************/
/***********************************************************************************/

using PortInf = std::pair<int, int>;
using PortSet = std::set<PortInf>;
using PortMap = std::map<PortInf, DataPtr>;

static PortSet allRequiredInputs(pugi::xml_node& ti) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:  static PortSet allRequiredInputs(pugi::xml_node& ti) {" << std::endl;
    PortSet res;  // duplicates are possible

    FOREACH_CHILD(p, ti.child("port_map"), "input") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      FOREACH_CHILD(p, ti.child('port_map'), 'input') {" << std::endl;
        int internal_layer_id = GetIntAttr(p, "internal_layer_id");
        int internal_port_id = GetIntAttr(p, "internal_port_id");
        res.emplace(internal_layer_id, internal_port_id);
    }
    FOREACH_CHILD(ec, ti.child("back_edges"), "edge") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      FOREACH_CHILD(ec, ti.child('back_edges'), 'edge') {" << std::endl;
        int to_layer_id = GetIntAttr(ec, "to-layer");
        int to_port_id = GetIntAttr(ec, "to-port");
        res.emplace(to_layer_id, to_port_id);
    }
    return res;
}

static PortSet allRequiredOutputs(pugi::xml_node& ti) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:  static PortSet allRequiredOutputs(pugi::xml_node& ti) {" << std::endl;
    PortSet res;  // duplicates are possible

    FOREACH_CHILD(p, ti.child("port_map"), "output") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      FOREACH_CHILD(p, ti.child('port_map'), 'output') {" << std::endl;
        int internal_layer_id = GetIntAttr(p, "internal_layer_id");
        int internal_port_id = GetIntAttr(p, "internal_port_id");
        res.emplace(internal_layer_id, internal_port_id);
    }
    FOREACH_CHILD(edge, ti.child("back_edges"), "edge") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      FOREACH_CHILD(edge, ti.child('back_edges'), 'edge') {" << std::endl;
        int to_layer_id = GetIntAttr(edge, "from-layer");
        int to_port_id = GetIntAttr(edge, "from-port");
        res.emplace(to_layer_id, to_port_id);
    }
    return res;
}

/***********************************************************************************/
/*******  Body Parser Helper  ******************************************************/
/***********************************************************************************/
using WBlob = TBlob<uint8_t>::Ptr;

class BodyParser {
public:
    BodyParser(pugi::xml_node& net_node, size_t ir_version, Precision prec)
        : body(net_node), parser(FormatParser(ir_version)), default_precision(prec) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:          : body(net_node), parser(FormatParser(ir_version)), default_precision(prec) {" << std::endl;}

    void parse(PortSet in_request, PortSet out_request) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      void parse(PortSet in_request, PortSet out_request) {" << std::endl;
        body.append_attribute("precision").set_value(default_precision.name());
        auto net = parser.Parse(body);

        for (const auto& pi : in_request) inputs[pi] = parser.GetDataBy(pi.first, pi.second);
        for (const auto& pi : out_request) outputs[pi] = parser.GetDataBy(pi.first, pi.second);

        // Mark data as network output. Just for check
        for (const auto& kvp : outputs) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:          for (const auto& kvp : outputs) {" << std::endl;
            auto& data = kvp.second;
            auto layer = data->getCreatorLayer().lock();
            auto& outs = layer->outData;
            auto o_idx = std::find(outs.begin(), outs.end(), data) - outs.begin();
            auto sts = net->addOutput(layer->name, o_idx, nullptr);
            IE_ASSERT(sts == OK) << "TI body. Cannot add output port for layer " << layer->name << " port index "
                                 << o_idx;
        }

        // Verify that all input/output are in use
        InputsDataMap in_info_map;
        std::map<std::string, DataPtr> out_info_map;
        net->getInputsInfo(in_info_map);
        net->getOutputsInfo(out_info_map);

        IE_ASSERT(in_info_map.size() == inputs.size()) << "TI body. There are unlinked inputs";
    }

    void setWeights(const WBlob& weights) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      void setWeights(const WBlob& weights) {" << std::endl;
        parser.SetWeights(weights);
    }

    const PortMap& getInsMap() const {
        return inputs;
    }
    const PortMap& getOutsMap() const {
        return outputs;
    }

private:
    pugi::xml_node& body;
    FormatParser parser;
    Precision default_precision;

    PortMap inputs;
    PortMap outputs;
};

CNNLayer::Ptr TILayerCreator::CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:  CNNLayer::Ptr TILayerCreator::CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) {" << std::endl;
    std::string ti_name = node.attribute("name").as_string();

    auto body = node.child("body");
    if (body.empty()) THROW_IE_EXCEPTION << "TensorIterator " << ti_name << " has no body";

    auto all_inputs = allRequiredInputs(node);
    auto all_outputs = allRequiredOutputs(node);

    auto parser = std::make_shared<BodyParser>(body, layerParsePrms.underIRVersion, layerParsePrms.prms.precision);
    parser->parse(all_inputs, all_outputs);

    auto ins = parser->getInsMap();
    auto outs = parser->getOutsMap();

    // fill in/outputs and map internal port to index
    std::map<PortInf, int> p2i;
    std::vector<DataPtr> inputs, outputs;
    for (const auto& p : all_inputs) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      for (const auto& p : all_inputs) {" << std::endl;
        IE_ASSERT(ins.find(p) != ins.end());
        p2i[p] = inputs.size();
        inputs.push_back(ins[p]);
    }
    for (const auto& p : all_outputs) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      for (const auto& p : all_outputs) {" << std::endl;
        IE_ASSERT(outs.find(p) != outs.end());
        p2i[p] = outputs.size();
        outputs.push_back(outs[p]);
    }

    // fill map external port to index
    std::map<int, int> e2i;
    {
        int in_indx = 0;
        FOREACH_CHILD(in, node.child("input"), "port") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:          FOREACH_CHILD(in, node.child('input'), 'port') {" << std::endl;
            int id = GetIntAttr(in, "id");
            e2i[id] = in_indx++;
        }
        int out_indx = 0;
        FOREACH_CHILD(in, node.child("output"), "port") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:          FOREACH_CHILD(in, node.child('output'), 'port') {" << std::endl;
            int id = GetIntAttr(in, "id");
            e2i[id] = out_indx++;
        }
    }

    std::vector<TensorIterator::PortMap> in_ports_maping, out_ports_maping, back_edges;

    auto parse_rule = [&](pugi::xml_node& pm) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      auto parse_rule = [&](pugi::xml_node& pm) {" << std::endl;
        int external_port_id = GetIntAttr(pm, "external_port_id");
        int internal_layer_id = GetIntAttr(pm, "internal_layer_id");
        int internal_port_id = GetIntAttr(pm, "internal_port_id");

        int axis = GetIntAttr(pm, "axis", -1);
        int stride = GetIntAttr(pm, "stride", 1);
        int part_size = GetIntAttr(pm, "part_size", 1);
        int start = GetIntAttr(pm, "start", 0);
        int end = GetIntAttr(pm, "end", -1);

        TensorIterator::PortMap res;
        res.from = e2i[external_port_id];
        res.to = p2i[{internal_layer_id, internal_port_id}];
        res.axis = axis;
        res.stride = stride;
        res.part_size = part_size;
        res.start = start;
        res.end = end;
        return res;
    };

    FOREACH_CHILD(pm, node.child("port_map"), "input") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      FOREACH_CHILD(pm, node.child('port_map'), 'input') {" << std::endl;
        in_ports_maping.push_back(parse_rule(pm));
    }
    FOREACH_CHILD(pm, node.child("port_map"), "output") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      FOREACH_CHILD(pm, node.child('port_map'), 'output') {" << std::endl;
        out_ports_maping.push_back(parse_rule(pm));
    }

    FOREACH_CHILD(ec, node.child("back_edges"), "edge") {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      FOREACH_CHILD(ec, node.child('back_edges'), 'edge') {" << std::endl;
        int from_l = GetIntAttr(ec, "from-layer");
        int from_p = GetIntAttr(ec, "from-port");
        int to_l = GetIntAttr(ec, "to-layer");
        int to_p = GetIntAttr(ec, "to-port");

        back_edges.push_back({p2i[{from_l, from_p}], p2i[{to_l, to_p}], -1, 1, 0, -1, 1});
    }

    // Hold parser as a shared_ptr into callback
    // Will be called outside to set weight blobs
    // for internal TI body layers
    layerParsePrms.internalWeightSet = [=](const WBlob& w) {
    std::cerr << "./inference-engine/src/inference_engine/ie_layer_parsers.cpp:      layerParsePrms.internalWeightSet = [=](const WBlob& w) {" << std::endl;
        parser->setWeights(w);
    };

    auto res = std::make_shared<TensorIterator>(layerParsePrms.prms);
    res->body.inputs = inputs;
    res->body.outputs = outputs;
    res->input_port_map = in_ports_maping;
    res->output_port_map = out_ports_maping;
    res->back_edges = back_edges;
    return res;
}

}  // namespace details
}  // namespace InferenceEngine
