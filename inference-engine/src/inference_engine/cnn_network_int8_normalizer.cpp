#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_int8_normalizer.hpp"

#include <data_stats.h>
#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <cassert>
#include <cmath>
#include <details/caseless.hpp>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "cnn_network_impl.hpp"
#include "cnn_network_stats_impl.hpp"
#include "debug.h"
#include "ie_util_internal.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

using StatsMap = std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>;

CNNStatisticHelper::CNNStatisticHelper(CNNNetwork& network,
                                       const std::map<std::string, NetworkNodeStatsPtr>& internalNodesStats,
                                       int maxSign, int maxUnsign) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                         int maxSign, int maxUnsign) {" << std::endl;
    internalNodesStats_ = internalNodesStats;
    network_ = network;
    maxSign_ = maxSign;
    maxUnsign_ = maxUnsign;

    NormalizeStatistic();
}

bool CNNStatisticHelper::canLayerBeQuantized(CNNLayer::Ptr layer) const {
    // verification of existing statistic for all inputs
    for (const auto i : layer->insData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (const auto i : layer->insData) {" << std::endl;
        if (internalNodesStats_.find(i.lock()->getCreatorLayer().lock()->name) == internalNodesStats_.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (internalNodesStats_.find(i.lock()->getCreatorLayer().lock()->name) == internalNodesStats_.end()) {" << std::endl;
            return false;
        }
    }
    // verification if there is a statistic for output of the layer
    if ((layer->outData.size() > 1) && (internalNodesStats_.find(layer->name) == internalNodesStats_.end())) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if ((layer->outData.size() > 1) && (internalNodesStats_.find(layer->name) == internalNodesStats_.end())) {" << std::endl;
        return false;
    }
    return true;
}

void CNNStatisticHelper::copyStatistics(const std::string& srcName, const std::string& dstName) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  void CNNStatisticHelper::copyStatistics(const std::string& srcName, const std::string& dstName) {" << std::endl;
    internalNodesStats_[dstName] = internalNodesStats_[srcName];
}

bool CNNStatisticHelper::hasNegativeOutput(const std::string& layerName, int outputPort) const {
    // TODO(amalyshe) parameter outputPort is not used yet, logic of dedication to the port
    // should be implemented

    NetworkNodeStatsPtr layerStat = internalNodesStats_.at(layerName);
    for (auto v : layerStat->_minOutputs) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto v : layerStat->_minOutputs) {" << std::endl;
        if (v < 0.f) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (v < 0.f) {" << std::endl;
            return true;
        }
    }
    return false;
}

InferenceEngine::Blob::Ptr CNNStatisticHelper::getInputScale(CNNLayer::Ptr layer) const {
    auto inDataPtr = layer->insData[0].lock();
    if (inDataPtr == nullptr)
        return nullptr;
    auto previousLayer = inDataPtr->getCreatorLayer().lock();
    std::string inputLayerName = previousLayer->name;

    // for case when we have the only average pooling before, we need to take this
    // statistic from input of avg pooling to compensate work of average pooling
    // and to stay in int8 as much as we can
    if (previousLayer->type == "Pooling" &&
        (previousLayer->precision == Precision::I8 || previousLayer->precision == Precision::U8)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          (previousLayer->precision == Precision::I8 || previousLayer->precision == Precision::U8)) {" << std::endl;
        // take input name to the pooling
        auto prevInDataPtr = previousLayer->insData[0].lock();
        if (prevInDataPtr == nullptr)
            return nullptr;
        inputLayerName = prevInDataPtr->getCreatorLayer().lock()->name;
    }
    size_t inputChannels = inDataPtr->getTensorDesc().getDims()[1];
    if (getStatistic(previousLayer)->_minOutputs.size() != inputChannels ||
        getStatistic(previousLayer)->_maxOutputs.size() != inputChannels) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          getStatistic(previousLayer)->_maxOutputs.size() != inputChannels) {" << std::endl;
        THROW_IE_EXCEPTION << "min and max sizes should be equal to input channels count for " << previousLayer->name;
    }

    // current normalization algorithm can have nodes with fp32 edges. it can happen only in places
    // of initial quantization of int8 chains. Currently adding scaleshift adds certain I8/U8 precision
    // but calcualtion of scales happens before adding of scale shifts.
    // for fixing problem with cases of not determined yet presision and for following of
    // quantizatoin scheme defined by normalizer, we are adding here verification of negative output
    // in some cases and then verify exact precision of I8/U8 on node for covering of fully determined cases
    int maxValue = hasNegativeOutput(previousLayer->name) ? maxSign_ : maxUnsign_;
    if (previousLayer->outData[0]->getPrecision() == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (previousLayer->outData[0]->getPrecision() == Precision::U8) {" << std::endl;
        maxValue = maxUnsign_;
    } else if (previousLayer->outData[0]->getPrecision() == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      } else if (previousLayer->outData[0]->getPrecision() == Precision::I8) {" << std::endl;
        maxValue = maxSign_;
    }

    return calculateScaleFactor(inputChannels, getStatistic(previousLayer), maxValue);
}

InferenceEngine::Blob::Ptr CNNStatisticHelper::getOutputScale(CNNLayer::Ptr layer) const {
    // TODO(amalyshe) for now we are looking to precision on the data node
    size_t outputChannels = layer->outData[0]->getTensorDesc().getDims()[1];
    if (layer->outData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->outData.size() != 1) {" << std::endl;
        THROW_IE_EXCEPTION << "Trying to get scales after layer having multiple output ports";
    }

    auto it = internalNodesStats_.find(layer->name);
    if (it == internalNodesStats_.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (it == internalNodesStats_.end()) {" << std::endl;
        return std::shared_ptr<Blob>();
    }

    if (getStatistic(layer)->_minOutputs.size() != outputChannels ||
        getStatistic(layer)->_maxOutputs.size() != outputChannels) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          getStatistic(layer)->_maxOutputs.size() != outputChannels) {" << std::endl;
        THROW_IE_EXCEPTION << "min and max sizes should be equal to output channels count for " << layer->name;
    }

    return calculateScaleFactor(outputChannels, getStatistic(layer),
                                layer->outData[0]->getPrecision() == Precision::I8 ? maxSign_ : maxUnsign_);
}

int CNNStatisticHelper::getMaxSignValue() const {
    return maxSign_;
}

InferenceEngine::Blob::Ptr CNNStatisticHelper::calculateScaleFactor(size_t channels, NetworkNodeStatsPtr stats,
                                                                    int maxInt) const {
    if (stats->_minOutputs.size() != channels || stats->_maxOutputs.size() != channels) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (stats->_minOutputs.size() != channels || stats->_maxOutputs.size() != channels) {" << std::endl;
        THROW_IE_EXCEPTION << "min and max sizes should be equal to channels count";
    }

    // Creating i-scale blob
    std::shared_ptr<Data> iScaleData =
        std::shared_ptr<Data>(new Data("scale", {Precision::FP32, {channels}, Layout::C}));
    auto iScale = CreateBlobFromData(iScaleData);
    iScale->allocate();
    float* iScaleMemory = static_cast<float*>(iScale->buffer());

    for (int c = 0; c < channels; c++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (int c = 0; c < channels; c++) {" << std::endl;
        // maxc = fmax(maxc, fabs(stats[k]->_minOutputs[c]));        // TODO Check if we should take minimums into
        // account
        float maxc = fabs(stats->_maxOutputs[c]);
        maxc = fmax(maxc, fabs(stats->_minOutputs[c]));

        iScaleMemory[c] = maxc / static_cast<float>(maxInt);

        if (fabs(iScaleMemory[c]) < 1e-7) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (fabs(iScaleMemory[c]) < 1e-7) {" << std::endl;
            iScaleMemory[c] = 1.0f;
        }
    }
    return iScale;
}

NetworkNodeStatsPtr CNNStatisticHelper::getStatistic(CNNLayer::Ptr layer) const {
    // TODO(amalyshe) all logic of traversing over network and get apropriate statistics should be here
    // for now it is a stub
    auto it = internalNodesStats_.find(getLatestInFuse(layer)->name);
    if (it != internalNodesStats_.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (it != internalNodesStats_.end()) {" << std::endl;
        return it->second;
    }
    THROW_IE_EXCEPTION << "no stat for layer " << getLatestInFuse(layer)->name;
}

CNNLayer::Ptr CNNStatisticHelper::getLatestInFuse(CNNLayer::Ptr layer) const {
    if (layer->outData[0]->getInputTo().size() == 1 &&
        (CaselessEq<std::string>()(layer->outData[0]->getInputTo().begin()->second->type, "relu") ||
         CNNNetworkInt8Normalizer::isReLULikeClamp(layer->outData[0]->getInputTo().begin()->second))) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:           CNNNetworkInt8Normalizer::isReLULikeClamp(layer->outData[0]->getInputTo().begin()->second))) {" << std::endl;
        return layer->outData[0]->getInputTo().begin()->second;
    }
    // Conv-Sum-ReLU fuse
    // We need to return original layer if it will be used as a sum parame and ReLU if
    // iterating over outputs of pointed layer and look for the only eltwise
    CNNLayer::Ptr eltwise = nullptr;
    if (layer->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->outData.size() == 1) {" << std::endl;
        for (auto it : layer->outData[0]->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (auto it : layer->outData[0]->getInputTo()) {" << std::endl;
            if (CaselessEq<std::string>()(it.second->type, "eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (CaselessEq<std::string>()(it.second->type, 'eltwise')) {" << std::endl;
                if (eltwise) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (eltwise) {" << std::endl;
                    THROW_IE_EXCEPTION << "Pattern when one layer pass data to several eltwise layers are not "
                                          "supported in int8 quantization";
                }
                eltwise = it.second;
            }
        }
    }

    if (eltwise) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (eltwise) {" << std::endl;
        // if current layer is not a convolution return it as finish of fuse
        if (!CaselessEq<std::string>()(layer->type, "convolution")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (!CaselessEq<std::string>()(layer->type, 'convolution')) {" << std::endl;
            return layer;
        } else {
            // look to the ports of eltwise
            if (eltwise->insData[0].lock() != nullptr
                    && eltwise->insData[1].lock() != nullptr
                    && eltwise->insData[1].lock()->getCreatorLayer().lock() == layer
                    && CaselessEq<std::string>()(eltwise->insData[0].lock()->getCreatorLayer().lock()->type, "convolution")
                    && eltwise->insData[0].lock()->getInputTo().size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      && eltwise->insData[0].lock()->getInputTo().size() == 1) {" << std::endl;
                // this is a case when two convolutions come to eltwise, the second one will be selected for fuse,
                // first will be used as sum operator
                return layer;
            }
            // given layer is a convolution and will be used for fuse, but we need to verify if there is ReLU after
            // eltwise
            if (eltwise->outData[0]->getInputTo().size() == 1 &&
                (CaselessEq<std::string>()(eltwise->outData[0]->getInputTo().begin()->second->type, "relu") ||
                 CNNNetworkInt8Normalizer::isReLULikeClamp(eltwise->outData[0]->getInputTo().begin()->second))) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                   CNNNetworkInt8Normalizer::isReLULikeClamp(eltwise->outData[0]->getInputTo().begin()->second))) {" << std::endl;
                return eltwise->outData[0]->getInputTo().begin()->second;
            }
            return eltwise;
        }
    }

    return layer;
}

void CNNStatisticHelper::NormalizeStatistic() {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  void CNNStatisticHelper::NormalizeStatistic() {" << std::endl;
    StatsMap newMap;

    // In case when we have statistics in negative range when min clamped value is 0,
    // we are changing statistics here to non negative. This is not fully correct behaviour since
    // it can extend range and affect accuracy, but this approach works quite well
    std::vector<CNNLayerPtr> sortedLayersRC = CNNNetSortTopologically(network_);
    for (auto l : sortedLayersRC) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto l : sortedLayersRC) {" << std::endl;
        if (CNNNetworkInt8Normalizer::isReLULikeClamp(l)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (CNNNetworkInt8Normalizer::isReLULikeClamp(l)) {" << std::endl;
            if (l->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (l->outData.size() == 1) {" << std::endl;
                size_t outputChannels = l->outData[0]->getTensorDesc().getDims()[1];
                auto oldStat = internalNodesStats_.find(l->name);
                if ((oldStat != internalNodesStats_.end()) && outputChannels > 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if ((oldStat != internalNodesStats_.end()) && outputChannels > 1) {" << std::endl;
                    for (size_t q = 0; q < oldStat->second->_minOutputs.size(); q++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      for (size_t q = 0; q < oldStat->second->_minOutputs.size(); q++) {" << std::endl;
                        oldStat->second->_minOutputs[q] = 0.f;
                    }
                }
            }
        }
    }

    float dummy = 0.0f;

    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network_);
    for (auto l : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto l : sortedLayers) {" << std::endl;
        // if layer's statistic exists in the newMap, ignore it
        if (newMap.find(l->name) != newMap.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (newMap.find(l->name) != newMap.end()) {" << std::endl;
            continue;
        }
        // verify if layer is starter layer for propagating of statistic
        bool isStarterLayer = false;

        // a case if we do not have converted statistic before the current layer
        // go over all inputs and verify if statistic exists for all of inputs
        bool allInputsHaveStatistics = true;
        for (auto i : l->insData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (auto i : l->insData) {" << std::endl;
            if (newMap.find(i.lock()->getCreatorLayer().lock()->name) == newMap.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (newMap.find(i.lock()->getCreatorLayer().lock()->name) == newMap.end()) {" << std::endl;
                allInputsHaveStatistics = false;
                break;
            }
        }
        // if we do not have statistic - verify who is consumer of this layer
        if (!allInputsHaveStatistics) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (!allInputsHaveStatistics) {" << std::endl;
            if (l->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (l->outData.size() == 1) {" << std::endl;
                for (auto it : l->outData[0]->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (auto it : l->outData[0]->getInputTo()) {" << std::endl;
                    if (CaselessEq<std::string>()(it.second->type, "scaleshift") ||
                        CaselessEq<std::string>()(it.second->type, "convolution") ||
                        CaselessEq<std::string>()(it.second->type, "fullyconnected")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          CaselessEq<std::string>()(it.second->type, 'fullyconnected')) {" << std::endl;
                        isStarterLayer = true;
                        break;
                    }
                }
            }
        } else {
            isStarterLayer = true;
        }
        if (CaselessEq<std::string>()(l->type, "scaleshift") || CaselessEq<std::string>()(l->type, "convolution") ||
            CaselessEq<std::string>()(l->type, "fullyconnected")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              CaselessEq<std::string>()(l->type, 'fullyconnected')) {" << std::endl;
            isStarterLayer = true;
        }

        if (!isStarterLayer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (!isStarterLayer) {" << std::endl;
            continue;
        }

        // we do not support yet layers for quantization which split data
        if (l->outData.size() != 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (l->outData.size() != 1) {" << std::endl;
            continue;
        }

        InferenceEngine::NetworkNodeStatsPtr currentStat = std::make_shared<NetworkNodeStats>();

        bool perChannelScale = true;

        if (CaselessEq<std::string>()(l->type, "concat") && l->outData.size() == 1 &&
            l->outData[0]->getTensorDesc().getDims().size() == 4 && allInputsHaveStatistics) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              l->outData[0]->getTensorDesc().getDims().size() == 4 && allInputsHaveStatistics) {" << std::endl;
            size_t concatLayerIdx = 0;
            for (int k = 0; k < l->insData.size(); k++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (int k = 0; k < l->insData.size(); k++) {" << std::endl;
                auto prevKLayer = l->insData[k].lock()->getCreatorLayer().lock();
                // looking for the statistic for prevKLayer
                auto kLayerStat = newMap.find(prevKLayer->name);
                if (kLayerStat != newMap.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (kLayerStat != newMap.end()) {" << std::endl;
                    for (size_t ikStat = 0; ikStat < kLayerStat->second->_maxOutputs.size();
                         ikStat++, concatLayerIdx++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                           ikStat++, concatLayerIdx++) {" << std::endl;
                        currentStat->_maxOutputs.push_back(kLayerStat->second->_maxOutputs[ikStat]);
                        currentStat->_minOutputs.push_back(kLayerStat->second->_minOutputs[ikStat]);
                    }
                } else {
                    THROW_IE_EXCEPTION << "We have incomplete statistic for predecessors of concat layer " << l->name;
                }
            }
        } else if (CaselessEq<std::string>()(l->type, "resample")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (CaselessEq<std::string>()(l->type, 'resample')) {" << std::endl;
            if (l->insData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (l->insData.size() == 1) {" << std::endl;
                CNNLayerPtr creator = l->insData[0].lock()->getCreatorLayer().lock();
                if (CaselessEq<std::string>()(creator->type, "concat")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (CaselessEq<std::string>()(creator->type, 'concat')) {" << std::endl;
                    auto concatStat = newMap[creator->name];
                    currentStat->_maxOutputs = concatStat->_maxOutputs;
                    currentStat->_minOutputs = concatStat->_minOutputs;
                    newMap[l->name] = currentStat;
                } else {
                    auto itOld = internalNodesStats_.find(l->name);
                    if (itOld != internalNodesStats_.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (itOld != internalNodesStats_.end()) {" << std::endl;
                        currentStat->_maxOutputs = itOld->second->_maxOutputs;
                        currentStat->_minOutputs = itOld->second->_minOutputs;
                        newMap[l->name] = currentStat;
                    }
                }
            }
        } else {
            // go over all children until we get convoluition, scaleshift, eltwise or unknown layer
            // layers Pooling and ReLU are passthrough
            // to understand the granularity of the scaling
            // layer concat is a layer which produce statistics and waterfall it down
            std::vector<CNNLayer::Ptr> toAnalyze;
            for (auto it : l->outData[0]->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (auto it : l->outData[0]->getInputTo()) {" << std::endl;
                toAnalyze.push_back(it.second);
            }

            if (CaselessEq<std::string>()(l->type, "eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (CaselessEq<std::string>()(l->type, 'eltwise')) {" << std::endl;
                perChannelScale = false;
            }
            while (!toAnalyze.empty() && perChannelScale) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              while (!toAnalyze.empty() && perChannelScale) {" << std::endl;
                CNNLayer::Ptr tl = toAnalyze.back();
                toAnalyze.pop_back();
                if (CaselessEq<std::string>()(tl->type, "pooling") || CaselessEq<std::string>()(tl->type, "relu") ||
                    CNNNetworkInt8Normalizer::isReLULikeClamp(tl) || CaselessEq<std::string>()(tl->type, "concat")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      CNNNetworkInt8Normalizer::isReLULikeClamp(tl) || CaselessEq<std::string>()(tl->type, 'concat')) {" << std::endl;
                    if (tl->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (tl->outData.size() == 1) {" << std::endl;
                        for (auto it : tl->outData[0]->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          for (auto it : tl->outData[0]->getInputTo()) {" << std::endl;
                            toAnalyze.push_back(it.second);
                        }
                    }
                } else if (CaselessEq<std::string>()(tl->type, "convolution")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  } else if (CaselessEq<std::string>()(tl->type, 'convolution')) {" << std::endl;
                    // verify number of groups
                    ConvolutionLayer* pConv = dynamic_cast<ConvolutionLayer*>(tl.get());
                    if (pConv == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (pConv == nullptr) {" << std::endl;
                        THROW_IE_EXCEPTION << "Layer " << tl->name << " is not instance of ConvolutionLayer class";
                    }
                    if (pConv->_group != pConv->_out_depth) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (pConv->_group != pConv->_out_depth) {" << std::endl;
                        perChannelScale = false;
                    }
                } else if (CaselessEq<std::string>()(tl->type, "eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  } else if (CaselessEq<std::string>()(tl->type, 'eltwise')) {" << std::endl;
                    perChannelScale = false;
                }
            }

            auto itOld = internalNodesStats_.find(getLatestInFuse(l)->name);
            if (itOld == internalNodesStats_.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (itOld == internalNodesStats_.end()) {" << std::endl;
                itOld = internalNodesStats_.find(l->name);
            }
            if (itOld != internalNodesStats_.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (itOld != internalNodesStats_.end()) {" << std::endl;
                if (!perChannelScale) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (!perChannelScale) {" << std::endl;
                    currentStat->_maxOutputs.resize(itOld->second->_maxOutputs.size());
                    if (!itOld->second->_maxOutputs.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (!itOld->second->_maxOutputs.empty()) {" << std::endl;
                        float max = FLT_MIN;
                        DataStats::GetDataAbsMax(&itOld->second->_maxOutputs[0], itOld->second->_maxOutputs.size(),
                                                 max);
                        std::fill(currentStat->_maxOutputs.begin(), currentStat->_maxOutputs.end(), max);
                    }

                    currentStat->_minOutputs.resize(itOld->second->_minOutputs.size());
                    if (!itOld->second->_minOutputs.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (!itOld->second->_minOutputs.empty()) {" << std::endl;
                        float min = FLT_MAX;
                        DataStats::GetDataMinMax(&itOld->second->_minOutputs[0], itOld->second->_minOutputs.size(), min,
                                                 dummy);
                        std::fill(currentStat->_minOutputs.begin(), currentStat->_minOutputs.end(), min);
                    }
                } else {
                    currentStat->_maxOutputs = itOld->second->_maxOutputs;
                    currentStat->_minOutputs = itOld->second->_minOutputs;
                }
            }

            if (l->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (l->outData.size() == 1) {" << std::endl;
                size_t ch_indx = l->outData[0]->getTensorDesc().getDims().size() > 1 ? 1 : 0;
                size_t outputChannels = l->outData[0]->getTensorDesc().getDims()[ch_indx];
                auto oldStat = internalNodesStats_.find(l->name);
                if ((oldStat != internalNodesStats_.end()) && outputChannels > 1 &&
                    oldStat->second->_minOutputs.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      oldStat->second->_minOutputs.size() == 1) {" << std::endl;
                    auto min = oldStat->second->_minOutputs[0];
                    auto max = oldStat->second->_maxOutputs[0];

                    currentStat->_minOutputs = std::vector<float>(outputChannels);
                    currentStat->_maxOutputs = std::vector<float>(outputChannels);
                    std::fill(currentStat->_minOutputs.begin(), currentStat->_minOutputs.end(), min);
                    std::fill(currentStat->_maxOutputs.begin(), currentStat->_maxOutputs.end(), max);
                }
            }
        }

        // propagate this statistic to all layers without scale in primitives
        if (!currentStat->_maxOutputs.empty() && !currentStat->_minOutputs.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (!currentStat->_maxOutputs.empty() && !currentStat->_minOutputs.empty()) {" << std::endl;
            std::vector<CNNLayer::Ptr> toAnalyze;
            toAnalyze.push_back(l);
            while (!toAnalyze.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              while (!toAnalyze.empty()) {" << std::endl;
                CNNLayer::Ptr tl = toAnalyze.back();
                toAnalyze.pop_back();
                newMap[tl->name] = currentStat;
                if (tl->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (tl->outData.size() == 1) {" << std::endl;
                    for (auto it : tl->outData[0]->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      for (auto it : tl->outData[0]->getInputTo()) {" << std::endl;
                        if (CaselessEq<std::string>()(it.second->type, "pooling") ||
                            CaselessEq<std::string>()(it.second->type, "relu") ||
                            CNNNetworkInt8Normalizer::isReLULikeClamp(it.second)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                              CNNNetworkInt8Normalizer::isReLULikeClamp(it.second)) {" << std::endl;
                            toAnalyze.push_back(it.second);
                        }
                    }
                }
            }
        }
    }

    internalNodesStats_ = newMap;
}

void CNNNetworkInt8Normalizer::AddLayerToCNNNetworkBeforeLayer(CNNLayer::Ptr newLayer, CNNLayer::Ptr successor,
                                                               size_t port) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                                                 size_t port) {" << std::endl;
    // verify if data exists
    if (newLayer && successor && successor->insData.size() > port) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (newLayer && successor && successor->insData.size() > port) {" << std::endl;
        // get the insData
        DataPtr pData = successor->insData[port].lock();

        Data* edge2 = new Data(*pData.get());
        DataPtr newEdge(edge2);
        newEdge->getInputTo().clear();
        newEdge->getInputTo()[successor->name] = successor;
        newEdge->setName(newLayer->name);
        newEdge->getCreatorLayer() = newLayer;
        successor->insData[port] = newEdge;
        newLayer->outData.push_back(newEdge);

        newLayer->insData.push_back(pData);
        pData->getInputTo().erase(successor->name);
        pData->getInputTo()[newLayer->name] = newLayer;
    } else {
        THROW_IE_EXCEPTION << "Invalid argument";
    }
}

CNNLayer::Ptr CNNNetworkInt8Normalizer::addU8ToI8Conversion(DataPtr data, CNNLayer::Ptr successor,
                                                            CNNStatisticHelper& statHelper) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                                              CNNStatisticHelper& statHelper) {" << std::endl;
    if (data->getPrecision() == Precision::U8 || data->getPrecision() == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (data->getPrecision() == Precision::U8 || data->getPrecision() == Precision::I8) {" << std::endl;
        size_t c = static_cast<size_t>(data->getDims()[1]);

        std::vector<float> ssWValues;
        std::vector<float> ssSValues;
        for (auto i = 0; i < c; i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (auto i = 0; i < c; i++) {" << std::endl;
            ssWValues.push_back(1.0f);
            ssSValues.push_back(0.0f);
        }
        std::string layerName = data->getCreatorLayer().lock()->name + "_Eltwise_ScaleShift_U8I8_" + successor->name;
        CNNLayer::Ptr newLayer = createDWConvolutionForScale(layerName, c, ssWValues.data(), ssSValues.data());
        newLayer->precision = Precision::I8;

        for (size_t i = 0; i < successor->insData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (size_t i = 0; i < successor->insData.size(); i++) {" << std::endl;
            if (successor->insData[i].lock() == data) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (successor->insData[i].lock() == data) {" << std::endl;
                AddLayerToCNNNetworkBeforeLayer(newLayer, successor, i);

                // update statistic to pass quantization smoothly
                if (newLayer->insData[0].lock() == nullptr)
                    continue;
                std::string inputLayerName = newLayer->insData[0].lock()->getCreatorLayer().lock()->name;
                statHelper.copyStatistics(inputLayerName, layerName);
                if (data->getPrecision() == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (data->getPrecision() == Precision::U8) {" << std::endl;
                    newLayer->outData[0]->setPrecision(Precision::I8);
                } else {
                    newLayer->outData[0]->setPrecision(Precision::U8);
                }
            }
        }
        return newLayer;
    }
    return nullptr;
}

void CNNNetworkInt8Normalizer::AddLayerToCNNNetworkAfterData(DataPtr pData, CNNLayer::Ptr layer,
                                                             const std::string& nextLayerName) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                                               const std::string& nextLayerName) {" << std::endl;
    // verify if data exists
    if (pData && layer && pData->getCreatorLayer().lock() &&
        pData->getInputTo().find(nextLayerName) != pData->getInputTo().end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          pData->getInputTo().find(nextLayerName) != pData->getInputTo().end()) {" << std::endl;
        CNNLayerPtr nextLayer = pData->getInputTo()[nextLayerName];

        DataPtr newEdgeAfterLayer(new Data(*pData.get()));
        newEdgeAfterLayer->setName(layer->name);
        newEdgeAfterLayer->getCreatorLayer() = layer;
        newEdgeAfterLayer->getInputTo().clear();
        newEdgeAfterLayer->getInputTo()[nextLayerName] = nextLayer;
        newEdgeAfterLayer->setPrecision(Precision::FP32);

        pData->getInputTo().erase(nextLayerName);
        pData->getInputTo()[layer->name] = layer;

        layer->insData.push_back(pData);
        layer->outData.push_back(newEdgeAfterLayer);

        for (size_t i = 0; i < nextLayer->insData.size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (size_t i = 0; i < nextLayer->insData.size(); i++) {" << std::endl;
            if (nextLayer->insData[i].lock() == pData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (nextLayer->insData[i].lock() == pData) {" << std::endl;
                nextLayer->insData[i] = newEdgeAfterLayer;
            }
        }
    } else {
        THROW_IE_EXCEPTION << "Invalid argument";
    }
}

void CNNNetworkInt8Normalizer::fillInScaleShift(ScaleShiftLayer* scshLayer, size_t c, float* weightsN,
                                                float* weightsD) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                                  float* weightsD) {" << std::endl;
    // Setting "scales"
    SizeVector weightsSize = {c};
    TensorDesc weightsDesc(Precision::FP32, weightsSize, InferenceEngine::C);
    scshLayer->_weights = InferenceEngine::make_shared_blob<float>(weightsDesc);
    scshLayer->_weights->allocate();
    float* weightsData = scshLayer->_weights->buffer();
    for (size_t i = 0; i < c; i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (size_t i = 0; i < c; i++) {" << std::endl;
        if (weightsN == nullptr && weightsD != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (weightsN == nullptr && weightsD != nullptr) {" << std::endl;
            weightsData[i] = 1.0 / weightsD[i];
        } else if (weightsD == nullptr && weightsN != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (weightsD == nullptr && weightsN != nullptr) {" << std::endl;
            weightsData[i] = weightsN[i];
        } else if (weightsN != nullptr && weightsD != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (weightsN != nullptr && weightsD != nullptr) {" << std::endl;
            weightsData[i] = weightsN[i] / weightsD[i];
        } else {
            weightsData[i] = 1.0;
        }
    }

    // Setting "shifts"
    SizeVector shiftsSize = {c};
    TensorDesc shiftsDesc(Precision::FP32, shiftsSize, InferenceEngine::C);
    scshLayer->_biases = InferenceEngine::make_shared_blob<float>(shiftsDesc);
    scshLayer->_biases->allocate();
    float* biasesData = scshLayer->_biases->buffer();
    for (size_t i = 0; i < c; i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (size_t i = 0; i < c; i++) {" << std::endl;
        biasesData[i] = 0.f;  // Setting to constant "0"
    }
}

void CNNNetworkInt8Normalizer::AddScaleShiftBetween(CNNNetwork& net, const CNNLayerPtr layer1, const CNNLayerPtr layer2,
                                                    CNNStatisticHelper& statHelper) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                                      CNNStatisticHelper& statHelper) {" << std::endl;
    if (CaselessEq<std::string>()(layer2->type, "priorbox") ||
        CaselessEq<std::string>()(layer2->type, "priorboxclustered")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          CaselessEq<std::string>()(layer2->type, 'priorboxclustered')) {" << std::endl;
        return;
    }

    // Searching the connection between the layers
    int l1_out_i = 0;
    for (; l1_out_i < layer1->outData.size(); l1_out_i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (; l1_out_i < layer1->outData.size(); l1_out_i++) {" << std::endl;
        if (layer1->outData[l1_out_i]->getInputTo().find(layer2->name) !=
            layer1->outData[l1_out_i]->getInputTo().end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              layer1->outData[l1_out_i]->getInputTo().end()) {" << std::endl;
            break;
        }
    }
    if (l1_out_i == layer1->outData.size()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (l1_out_i == layer1->outData.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Can't find layer " << layer2->name << " among layer " << layer1->name << " outputs";
    }

    int l2_in_i = 0;
    for (; l2_in_i < layer2->insData.size(); l2_in_i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (; l2_in_i < layer2->insData.size(); l2_in_i++) {" << std::endl;
        if (layer2->insData[l2_in_i].lock() != nullptr
                && layer2->insData[l2_in_i].lock()->getCreatorLayer().lock() == layer1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  && layer2->insData[l2_in_i].lock()->getCreatorLayer().lock() == layer1) {" << std::endl;
            break;
        }
    }
    if (l2_in_i == layer2->insData.size()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (l2_in_i == layer2->insData.size()) {" << std::endl;
        THROW_IE_EXCEPTION << "Can't find layer " << layer2->name << " among layer " << layer1->name << " inputs";
    }

    DataPtr outData = layer1->outData[l1_out_i];

    Blob::Ptr oScaleBlob = nullptr;
    if (layer1->blobs.find("o-scale") != layer1->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer1->blobs.find('o-scale') != layer1->blobs.end()) {" << std::endl;
        oScaleBlob = layer1->blobs["o-scale"];
    }

    Blob::Ptr iScaleBlob = nullptr;
    if (layer2->blobs.find("i-scale") != layer2->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer2->blobs.find('i-scale') != layer2->blobs.end()) {" << std::endl;
        iScaleBlob = layer2->blobs["i-scale"];
    }

    if (iScaleBlob == nullptr && oScaleBlob == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (iScaleBlob == nullptr && oScaleBlob == nullptr) {" << std::endl;
        return;  // No multipliers found around this edge. We can't create a ScaleShift here;
    } else {
        // Creating a ScaleShiftLayer
        std::string prefix;
        float *iScaleBuffer = nullptr, *oScaleBuffer = nullptr;
        if (oScaleBlob != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (oScaleBlob != nullptr) {" << std::endl;
            oScaleBuffer = static_cast<float*>(oScaleBlob->buffer());
            prefix += "o";
        }
        if (iScaleBlob != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (iScaleBlob != nullptr) {" << std::endl;
            iScaleBuffer = static_cast<float*>(iScaleBlob->buffer());
            prefix += "i";
        }

        std::string layerName = layer1->name + "_" + prefix + "ScaleShift_" + layer2->name;
        LayerParams ssCnnLayerParams {layerName, "ScaleShift", Precision::FP32};
        CNNLayerPtr ssCnnLayer(new ScaleShiftLayer(ssCnnLayerParams));

        AddLayerToCNNNetworkAfterData(outData, ssCnnLayer, layer2->name);

        size_t c = static_cast<size_t>(outData->getDims()[1]);

        {
            ScaleShiftLayer* scshLayer = dynamic_cast<ScaleShiftLayer*>(ssCnnLayer.get());
            if (scshLayer == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (scshLayer == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "Layer " << ssCnnLayer->name << " is not instance of ScaleShiftLayer class";
            }
            fillInScaleShift(scshLayer, c, oScaleBuffer, iScaleBuffer);
        }

        Precision odPrecision = Precision::FP32;
        if (layer2->precision == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (layer2->precision == Precision::I8) {" << std::endl;
            odPrecision = statHelper.hasNegativeOutput(layer1->name) ? Precision::I8 : Precision::U8;
        }
        ssCnnLayer->outData[0]->setPrecision(odPrecision);
    }
}

void CNNNetworkInt8Normalizer::AddScaleShifts(CNNNetwork& net, CNNStatisticHelper& statHelper) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  void CNNNetworkInt8Normalizer::AddScaleShifts(CNNNetwork& net, CNNStatisticHelper& statHelper) {" << std::endl;
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>> pairs;

    for (auto iter : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto iter : sortedLayers) {" << std::endl;
        for (int l1_out_i = 0; l1_out_i < iter->outData.size(); l1_out_i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (int l1_out_i = 0; l1_out_i < iter->outData.size(); l1_out_i++) {" << std::endl;
            for (auto nextIter : iter->outData[l1_out_i]->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (auto nextIter : iter->outData[l1_out_i]->getInputTo()) {" << std::endl;
                CNNLayer::Ptr next = nextIter.second;

                // Checking for an INT8 convolution or fully connected with FP32 output
                if ((CaselessEq<std::string>()(iter->type, "Convolution") ||
                     CaselessEq<std::string>()(iter->type, "FullyConnected")) &&
                    iter->precision == Precision::I8 && next->precision == Precision::FP32 &&
                    iter->outData[l1_out_i]->getPrecision() == Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      iter->outData[l1_out_i]->getPrecision() == Precision::FP32) {" << std::endl;
                    // Do nothing here only if iter provides data to fp32 layers
                    // MKLDNNPlugin will generate x8->f32 convolution

                } else if ((iter->precision != Precision::FP32 && next->precision == Precision::FP32) ||
                           (iter->precision == Precision::FP32 && next->precision != Precision::FP32)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                             (iter->precision == Precision::FP32 && next->precision != Precision::FP32)) {" << std::endl;
                    pairs.push_back(std::pair<CNNLayerPtr, CNNLayerPtr>(iter, next));
                }
            }
        }
    }

    for (auto& pair : pairs) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto& pair : pairs) {" << std::endl;
        AddScaleShiftBetween(net, pair.first, pair.second, statHelper);
    }
}

void CNNNetworkInt8Normalizer::ClampsToReLU(CNNNetwork& net, CNNStatisticHelper& statHelper) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  void CNNNetworkInt8Normalizer::ClampsToReLU(CNNNetwork& net, CNNStatisticHelper& statHelper) {" << std::endl;
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    for (auto iter : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto iter : sortedLayers) {" << std::endl;
        if (isReLULikeClamp(iter) && (iter->precision == Precision::I8 || iter->precision == Precision::U8)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (isReLULikeClamp(iter) && (iter->precision == Precision::I8 || iter->precision == Precision::U8)) {" << std::endl;
            std::string layerName = iter->name + "_ReLU";
            LayerParams ssCnnLayerParams {layerName, "ReLU", iter->precision};
            CNNLayerPtr ssCnnLayer(new ReLULayer(ssCnnLayerParams));

            auto previousLayer = iter->insData[0].lock()->getCreatorLayer().lock();
            ssCnnLayer->insData.push_back(iter->insData[0]);
            if (ssCnnLayer->insData[0].lock() == nullptr)
                continue;
            ssCnnLayer->insData[0].lock()->getInputTo().erase(iter->name);
            ssCnnLayer->insData[0].lock()->getInputTo()[iter->name] = ssCnnLayer;

            ssCnnLayer->outData.push_back(iter->outData[0]);
            ssCnnLayer->outData[0]->getCreatorLayer() = ssCnnLayer;

            iter->insData.clear();
            iter->outData.clear();
        }
    }
}

void CNNNetworkInt8Normalizer::ScaleDataToInt(const float* srcData, size_t srcSize, Blob::Ptr int8blob,
                                              const std::vector<float>& scales) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                                const std::vector<float>& scales) {" << std::endl;
    if (scales.size() == 0 || /*srcblob->size()*/ srcSize % scales.size() != 0) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (scales.size() == 0 || /*srcblob->size()*/ srcSize % scales.size() != 0) {" << std::endl;
        THROW_IE_EXCEPTION << "Wrong number of scale factors";
    }

    size_t channels = scales.size();
    size_t channelSize = /*srcblob->size()*/ srcSize / channels;

    const float* data = srcData;
    if (int8blob->getTensorDesc().getPrecision() == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (int8blob->getTensorDesc().getPrecision() == Precision::I8) {" << std::endl;
        int8_t* int8data = static_cast<int8_t*>(int8blob->buffer());
        int minValue = std::numeric_limits<int8_t>::min();
        int maxValue = std::numeric_limits<int8_t>::max();

        size_t offset;

        float val;

        for (size_t ch = 0; ch < channels; ch++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (size_t ch = 0; ch < channels; ch++) {" << std::endl;
            offset = channelSize * ch;

            for (size_t i = 0; i < channelSize; i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (size_t i = 0; i < channelSize; i++) {" << std::endl;
                val = data[offset + i] * scales[ch];

                if (val > maxValue) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (val > maxValue) {" << std::endl;
                    val = maxValue;
                } else if (val < minValue) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  } else if (val < minValue) {" << std::endl;
                    val = minValue;
                }

                int8data[offset + i] = round(val);
            }
        }
    } else if (int8blob->getTensorDesc().getPrecision() == Precision::I32) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      } else if (int8blob->getTensorDesc().getPrecision() == Precision::I32) {" << std::endl;
        int32_t* int32data = static_cast<int32_t*>(int8blob->buffer());
        int maxValue = std::numeric_limits<int32_t>::max();
        int minValue = std::numeric_limits<int32_t>::min();

        size_t offset;

        float val;

        for (size_t ch = 0; ch < channels; ch++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (size_t ch = 0; ch < channels; ch++) {" << std::endl;
            offset = channelSize * ch;

            for (size_t i = 0; i < channelSize; i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (size_t i = 0; i < channelSize; i++) {" << std::endl;
                val = data[offset + i] * scales[ch];

                if (val > maxValue) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (val > maxValue) {" << std::endl;
                    val = maxValue;
                } else if (val < minValue) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  } else if (val < minValue) {" << std::endl;
                    val = minValue;
                }

                int32data[offset + i] = round(val);
            }
        }
    }
}

CNNLayer::Ptr CNNNetworkInt8Normalizer::createDWConvolutionForScale(const std::string& layerName, size_t channels,
                                                                    float* ssWValues, float* ssSValues) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                                                      float* ssWValues, float* ssSValues) {" << std::endl;
    // create new Convolution layer
    LayerParams params;
    params.name = layerName;
    params.precision = Precision::FP32;
    params.type = "Convolution";

    CNNLayerPtr lptr = std::make_shared<ConvolutionLayer>(params);
    auto* pConv = dynamic_cast<ConvolutionLayer*>(lptr.get());
    if (pConv == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (pConv == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << "Layer " << lptr->name << " is not instance of ConvolutionLayer class";
    }

    pConv->_kernel.insert(X_AXIS, 1);
    pConv->_kernel.insert(Y_AXIS, 1);
    pConv->_stride.insert(X_AXIS, 1);
    pConv->_stride.insert(Y_AXIS, 1);
    pConv->_padding.insert(X_AXIS, 0);
    pConv->_padding.insert(Y_AXIS, 0);
    pConv->_pads_end.insert(X_AXIS, 0);
    pConv->_pads_end.insert(Y_AXIS, 0);
    pConv->_dilation.insert(X_AXIS, 1);
    pConv->_dilation.insert(Y_AXIS, 1);

    pConv->_out_depth = channels;
    // mkl-dnn does not have i8 depthwise convolution accepting signed i8 input
    // when it is available, need to uncomment below lines

    // workaround - creation of new weights for simple convolution
    if (pConv->_out_depth % 16 == 0) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (pConv->_out_depth % 16 == 0) {" << std::endl;
        pConv->_group = pConv->_out_depth / 16;
        Blob::Ptr weights = nullptr;
        std::shared_ptr<Data> wData =
            std::shared_ptr<Data>(new Data("weights", {Precision::FP32, {pConv->_out_depth * 16}, Layout::C}));
        weights = CreateBlobFromData(wData);
        weights->allocate();
        float* buffer = weights->buffer().as<float*>();
        size_t iDist = 0, iSrc = 0;
        for (size_t g = 0; g < pConv->_group; g++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (size_t g = 0; g < pConv->_group; g++) {" << std::endl;
            for (size_t k = 0; k < 16; k++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (size_t k = 0; k < 16; k++) {" << std::endl;
                for (size_t s = 0; s < 16; s++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (size_t s = 0; s < 16; s++) {" << std::endl;
                    buffer[iDist++] = (s == k) ? ssWValues[iSrc++] : 0.f;
                }
            }
        }
        pConv->_weights = weights;
        pConv->blobs["weights"] = weights;
    } else {
        Blob::Ptr weights = nullptr;
        std::shared_ptr<Data> wData = std::shared_ptr<Data>(
            new Data("weights", {Precision::FP32, {pConv->_out_depth * pConv->_out_depth}, Layout::C}));
        weights = CreateBlobFromData(wData);
        weights->allocate();
        float* buffer = weights->buffer().as<float*>();
        for (size_t i = 0, idx = 0; i < pConv->_out_depth; i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (size_t i = 0, idx = 0; i < pConv->_out_depth; i++) {" << std::endl;
            for (size_t j = 0; j < pConv->_out_depth; j++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (size_t j = 0; j < pConv->_out_depth; j++) {" << std::endl;
                if (i == j) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (i == j) {" << std::endl;
                    buffer[idx] = ssWValues[i];
                } else {
                    buffer[idx] = 0.f;
                }
                idx++;
            }
        }
        pConv->_weights = weights;
        pConv->blobs["weights"] = weights;
        pConv->_group = 1;
    }
    // end of workaround

    // fililng of biases
    Blob::Ptr biasesBlob = nullptr;
    std::shared_ptr<Data> bData =
        std::shared_ptr<Data>(new Data("biases", {Precision::FP32, {pConv->_out_depth}, Layout::C}));
    biasesBlob = CreateBlobFromData(bData);
    biasesBlob->allocate();
    float* bufferBiases = biasesBlob->buffer().as<float*>();
    for (size_t c = 0; c < pConv->_out_depth; c++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (size_t c = 0; c < pConv->_out_depth; c++) {" << std::endl;
        bufferBiases[c] = ssSValues[c];
    }
    pConv->_biases = biasesBlob;

    pConv->blobs["weights"] = pConv->_weights;
    pConv->blobs["biases"] = pConv->_biases;
    return lptr;
}

void CNNNetworkInt8Normalizer::replaceScaleShiftByDWConvolution(CNNNetwork& net) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  void CNNNetworkInt8Normalizer::replaceScaleShiftByDWConvolution(CNNNetwork& net) {" << std::endl;
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);
    for (auto layer : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto layer : sortedLayers) {" << std::endl;
        if (CaselessEq<std::string>()(layer->type, "scaleshift") &&
            layer->insData[0].lock()->getCreatorLayer().lock() &&
            !CaselessEq<std::string>()(layer->insData[0].lock()->getCreatorLayer().lock()->type, "input") &&
            layer->outData[0]->getInputTo().size() > 0) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              layer->outData[0]->getInputTo().size() > 0) {" << std::endl;
            const auto dims = layer->insData[0].lock()->getTensorDesc().getDims();
            // only four or five dimensions Convolution layers are supported
            if ((dims.size() == 4) || (dims.size() == 5)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if ((dims.size() == 4) || (dims.size() == 5)) {" << std::endl;
                // verification if this layer does not pass data to PriorBox, if it passes, we do not substitute
                bool notToPriorBox = true;
                for (auto o : layer->outData[0]->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (auto o : layer->outData[0]->getInputTo()) {" << std::endl;
                    if (CaselessEq<std::string>()(o.second->type, "priorbox") ||
                        CaselessEq<std::string>()(o.second->type, "priorboxclustered")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          CaselessEq<std::string>()(o.second->type, 'priorboxclustered')) {" << std::endl;
                        notToPriorBox = false;
                    }
                }
                if (notToPriorBox) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (notToPriorBox) {" << std::endl;
                    ScaleShiftLayer* pSS = dynamic_cast<ScaleShiftLayer*>(layer.get());
                    float* ssWValues = pSS->_weights->buffer().as<float*>();
                    float* ssSValues = pSS->_biases->buffer().as<float*>();
                    CNNLayer::Ptr newLayer = createDWConvolutionForScale(
                        layer->name, layer->outData[0]->getTensorDesc().getDims()[1], ssWValues, ssSValues);

                    newLayer->outData = layer->outData;
                    newLayer->outData[0]->getCreatorLayer() = newLayer;
                    newLayer->insData = layer->insData;
                    if (newLayer->insData[0].lock() == nullptr)
                        continue;
                    newLayer->insData[0].lock()->getInputTo().erase(layer->name);
                    newLayer->insData[0].lock()->getInputTo()[newLayer->name] = newLayer;
                }
            }
        }
    }
}

void CNNNetworkInt8Normalizer::QuantizeConvolutionOrFullyConnected(CNNLayer::Ptr target_layer,
                                                                   CNNStatisticHelper& statHelper) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                                                     CNNStatisticHelper& statHelper) {" << std::endl;
    size_t inputChannels = target_layer->insData[0].lock()->getTensorDesc().getDims()[1];
    size_t outputChannels = target_layer->outData[0]->getTensorDesc().getDims()[1];

    auto iScale = statHelper.getInputScale(target_layer);
    if (iScale == nullptr)
        THROW_IE_EXCEPTION << "Layer '" << target_layer->name << "'has invalid scale";

    target_layer->blobs["i-scale"] = iScale;

    Blob::Ptr weights = nullptr;
    Blob::Ptr biases = nullptr;

    Blob::Ptr int8weights = nullptr;
    Blob::Ptr int32biases = nullptr;

    if (target_layer->blobs.find("weights") != target_layer->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (target_layer->blobs.find('weights') != target_layer->blobs.end()) {" << std::endl;
        weights = target_layer->blobs["weights"];

        // Creating int8 weights blob
        std::shared_ptr<Data> int8WeightsData =
            std::shared_ptr<Data>(new Data("weights", TensorDesc(Precision::I8, weights->getTensorDesc().getDims(),
                                                                 weights->getTensorDesc().getLayout())));
        int8weights = CreateBlobFromData(int8WeightsData);
        int8weights->allocate();
        target_layer->blobs["weights"] = int8weights;
    }

    if (target_layer->blobs.find("biases") != target_layer->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (target_layer->blobs.find('biases') != target_layer->blobs.end()) {" << std::endl;
        biases = target_layer->blobs["biases"];

        // Creating int8 biases blob
        std::shared_ptr<Data> int32BiasesData =
            std::shared_ptr<Data>(new Data("biases", TensorDesc(Precision::I32, biases->getTensorDesc().getDims(),
                                                                biases->getTensorDesc().getLayout())));
        int32biases = CreateBlobFromData(int32BiasesData);
        int32biases->allocate();
        target_layer->blobs["biases"] = int32biases;
    }

    std::vector<float> weightScalers;

    // Creating w-scale blob
    if (weights) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (weights) {" << std::endl;
        const float* weight = static_cast<const float*>(weights->buffer());

        WeightableLayer* pConv = dynamic_cast<WeightableLayer*>(target_layer.get());
        ConvolutionLayer* pConv1 = dynamic_cast<ConvolutionLayer*>(target_layer.get());

        if (pConv1 != nullptr && pConv1->_group == 0) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (pConv1 != nullptr && pConv1->_group == 0) {" << std::endl;
            THROW_IE_EXCEPTION << "Convolution '" << target_layer->name << "'has wrong groups number == 0";
        }
        int group = 1;
        if (pConv1 != nullptr && pConv1->_group != 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (pConv1 != nullptr && pConv1->_group != 1) {" << std::endl;
            group = pConv1->_group;
        }

        std::vector<float> newWeights;  // "new" weights are weights multiplied by i-scale

        size_t W_CO = outputChannels / group, W_CI = inputChannels / group,
               W_HW = weights->size() / W_CI / W_CO / group;

        {
            float* iScaleMemory = static_cast<float*>(iScale->buffer());
            for (size_t g = 0; g < group; g++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (size_t g = 0; g < group; g++) {" << std::endl;
                for (size_t co = 0; co < W_CO; co++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (size_t co = 0; co < W_CO; co++) {" << std::endl;
                    for (size_t ci = 0; ci < W_CI; ci++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      for (size_t ci = 0; ci < W_CI; ci++) {" << std::endl;
                        size_t kernelBase = g * W_CO * W_CI * W_HW + co * W_CI * W_HW + ci * W_HW;
                        for (size_t hw = 0; hw < W_HW; hw++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          for (size_t hw = 0; hw < W_HW; hw++) {" << std::endl;
                            newWeights.push_back(weight[kernelBase + hw] * iScaleMemory[g * W_CI + ci]);
                        }
                    }
                }
            }
        }
        if (newWeights.empty())
            THROW_IE_EXCEPTION << "Could not quantize layer '" << target_layer->name << "'. Invalid layer parameters.";
        size_t outChannelSize = weights->getTensorDesc().getDims().back() / W_CO / group;

        // Calculating weights normalization scale factor (w-scale)

        std::set<double> individualsG;
        size_t co;
        float* weight_convolution;
        bool bwquantized = false;
        double symQuant = 0.f;

        for (co = 0, weight_convolution = &newWeights[0]; co < outputChannels;
             co++, weight_convolution += outChannelSize) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:               co++, weight_convolution += outChannelSize) {" << std::endl;
            for (size_t i = 0; i < outChannelSize && individualsG.size() < 256; i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (size_t i = 0; i < outChannelSize && individualsG.size() < 256; i++) {" << std::endl;
                individualsG.insert(static_cast<double>(weight_convolution[i]));
            }
        }
        // If we have 256 quantums for all filters in convolution, it can be already int8 quantized weights
        // We can support symmetric quantization
        // Below conditions verify if weights are symmetric quantized around 0, what are min/max borders
        // These parameters are required to repeat exactly the same quantum as model was trained
        // The algorithm of restoring min/max parameters has couple assumptions which might not work for 100%
        // cases. We want to explicitly define them. We assume that
        // 1. All convolutions have 1st quantum either from positive or negative side. See how we calculate symQuant
        // 2. If quantization is not symmetric, there should be quant on one of the side which demonstrate this
        if (individualsG.size() < 256) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (individualsG.size() < 256) {" << std::endl;
            // going over weights and verify that weights stay on quant positions
            std::set<double> intervals;
            double prev = 0.f;
            for (auto it = individualsG.begin(); it != individualsG.end(); it++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (auto it = individualsG.begin(); it != individualsG.end(); it++) {" << std::endl;
                if (prev) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (prev) {" << std::endl;
                    intervals.insert(*it - prev);
                }
                prev = *it;
            }
            symQuant = *(intervals.begin());
            std::set<double> divs;
            prev = 0.f;
            for (auto it = individualsG.begin(); it != individualsG.end(); it++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (auto it = individualsG.begin(); it != individualsG.end(); it++) {" << std::endl;
                if (prev) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (prev) {" << std::endl;
                    divs.insert((*it - prev) / symQuant);
                }
                prev = *it;
            }

            bwquantized = true;
            for (auto it3 = divs.begin(); it3 != divs.end(); it3++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (auto it3 = divs.begin(); it3 != divs.end(); it3++) {" << std::endl;
                if (fabs(round(*it3) - *it3) > 0.001) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (fabs(round(*it3) - *it3) > 0.001) {" << std::endl;
                    bwquantized = false;
                }
            }

            // we want to make sure that quantization is symmetric. this way we are looking for the
            // value in weights matching to the quant (positive or negative
            if (bwquantized) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (bwquantized) {" << std::endl;
                // take the minimal and maximum values on calculated symQuant and compare with data from individuals
                double minCalc = symQuant * -128.0f;
                double maxCalc = symQuant * 128.0f;
                for (auto it = individualsG.begin(); it != individualsG.end(); it++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (auto it = individualsG.begin(); it != individualsG.end(); it++) {" << std::endl;
                    if (*it < minCalc || *it > maxCalc) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (*it < minCalc || *it > maxCalc) {" << std::endl;
                        bwquantized = false;
                    }
                }
            }
        }
        if (bwquantized && symQuant != 0.0f) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (bwquantized && symQuant != 0.0f) {" << std::endl;
            float max = symQuant * 127.0f;
            for (co = 0, weight_convolution = &newWeights[0]; co < outputChannels;
                 co++, weight_convolution += outChannelSize) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                   co++, weight_convolution += outChannelSize) {" << std::endl;
                float scaler = static_cast<float>(statHelper.getMaxSignValue()) / max;
                weightScalers.push_back(scaler);
            }
        } else {
            for (co = 0, weight_convolution = &newWeights[0]; co < outputChannels;
                 co++, weight_convolution += outChannelSize) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                   co++, weight_convolution += outChannelSize) {" << std::endl;
                float max = FLT_MIN;
                DataStats::GetDataAbsMax(weight_convolution, outChannelSize, max);

                float scaler = static_cast<float>(statHelper.getMaxSignValue()) / max;
                weightScalers.push_back(scaler);
            }
        }

        std::shared_ptr<Data> wScaleData =
            std::shared_ptr<Data>(new Data("w-scale", {Precision::FP32, {outputChannels}, Layout::C}));
        auto wScale = CreateBlobFromData(wScaleData);
        wScale->allocate();

        float* wScaleMemory = static_cast<float*>(wScale->buffer());

        for (size_t i = 0; i < outputChannels; i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          for (size_t i = 0; i < outputChannels; i++) {" << std::endl;
            wScaleMemory[i] = 1.0 / weightScalers[i];
        }
        target_layer->blobs["w-scale"] = wScale;

        auto oScale = statHelper.getOutputScale(statHelper.getLatestInFuse(target_layer));
        if (oScale) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (oScale) {" << std::endl;
            // there might not be o-scale if we do not have statistic after convolution that means
            // returning to float precision after convolution
            target_layer->blobs["o-scale"] = oScale;

            // debug scales. Need to compare with actual values in FP32 scoring
            target_layer->blobs["ext-scale"] = target_layer->blobs["o-scale"];
        } else {
            // we do not have statistics here, we cannot calculate requantizatin scales,
            // next layer will be calculated in fp32
            // it's time to return forcedly edge to fp32 as well
            target_layer->outData[0]->setPrecision(Precision::FP32);
        }

        // Normalizing the weights
        ScaleDataToInt(&newWeights[0], weights->size(), int8weights, weightScalers);
    }

    // Normalizing the biases
    if (biases) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (biases) {" << std::endl;
        const float* bias = static_cast<const float*>(biases->buffer());
        ScaleDataToInt(bias, biases->size(), int32biases, weightScalers);
    }
}

bool CNNNetworkInt8Normalizer::layerProducesFloat(const CNNLayer::Ptr layer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  bool CNNNetworkInt8Normalizer::layerProducesFloat(const CNNLayer::Ptr layer) {" << std::endl;
    // currently we support only case of layers which have one output port
    if (layer->outData.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->outData.size() > 1) {" << std::endl;
        return false;
    }

    bool consumersFP32 = true;
    for (const auto dOut : layer->outData[0]->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (const auto dOut : layer->outData[0]->getInputTo()) {" << std::endl;
        if (dOut.second->precision != Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (dOut.second->precision != Precision::FP32) {" << std::endl;
            consumersFP32 = false;
        }
    }
    return consumersFP32;
}

void CNNNetworkInt8Normalizer::returnTailToFP32(const CNNLayer::Ptr layer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  void CNNNetworkInt8Normalizer::returnTailToFP32(const CNNLayer::Ptr layer) {" << std::endl;
    std::set<CNNLayer::Ptr> layersToReturn;
    if (layerProducesFloat(layer)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layerProducesFloat(layer)) {" << std::endl;
        layersToReturn.insert(layer);
    }

    while (!layersToReturn.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      while (!layersToReturn.empty()) {" << std::endl;
        CNNLayer::Ptr layerA = *layersToReturn.begin();
        layersToReturn.erase(layerA);
        // 1. if it is Pooling layer, or concat layer, we can return it to FP32 as well
        // we need to return it's out data
        if ((CaselessEq<std::string>()(layerA->type, "pooling") || CaselessEq<std::string>()(layerA->type, "concat")) &&
            layerA->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              layerA->outData.size() == 1) {" << std::endl;
            layerA->precision = Precision::FP32;
            layerA->outData[0]->setPrecision(Precision::FP32);
        }

        if ((CaselessEq<std::string>()(layerA->type, "convolution") ||
             CaselessEq<std::string>()(layerA->type, "fullyconnected") ||
             CaselessEq<std::string>()(layerA->type, "relu") || isReLULikeClamp(layerA)) &&
            layerA->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              layerA->outData.size() == 1) {" << std::endl;
            layerA->outData[0]->setPrecision(Precision::FP32);
            if (CaselessEq<std::string>()(layerA->type, "relu")
                    && layerA->insData[0].lock() != nullptr
                    && canLayerBeI8(layerA->insData[0].lock()->getCreatorLayer().lock())) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      && canLayerBeI8(layerA->insData[0].lock()->getCreatorLayer().lock())) {" << std::endl;
                layerA->precision = Precision::FP32;
                layerA->insData[0].lock()->getCreatorLayer().lock()->outData[0]->setPrecision(Precision::FP32);
            }
        }

        // adding parents for analysis
        if (!CaselessEq<std::string>()(layerA->type, "convolution") &&
            !CaselessEq<std::string>()(layerA->type, "fullyconnected")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              !CaselessEq<std::string>()(layerA->type, 'fullyconnected')) {" << std::endl;
            // for all parents, if they produce data to only FP32 layers
            for (auto i : layerA->insData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (auto i : layerA->insData) {" << std::endl;
                DataPtr d = i.lock();
                if (d != nullptr && d->getCreatorLayer().lock()->precision != Precision::FP32 &&
                    (CaselessEq<std::string>()(layerA->type, "pooling") ||
                     CaselessEq<std::string>()(layerA->type, "relu") || isReLULikeClamp(layerA) ||
                     CaselessEq<std::string>()(layerA->type, "concat"))) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                       CaselessEq<std::string>()(layerA->type, 'concat'))) {" << std::endl;
                    if (layerProducesFloat(d->getCreatorLayer().lock())) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (layerProducesFloat(d->getCreatorLayer().lock())) {" << std::endl;
                        layersToReturn.insert(d->getCreatorLayer().lock());
                    }
                }
            }
        }
    }
}

bool CNNNetworkInt8Normalizer::canLayerBeI8(const CNNLayer::Ptr& layer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  bool CNNNetworkInt8Normalizer::canLayerBeI8(const CNNLayer::Ptr& layer) {" << std::endl;
    // fusion can happen only if initial layer supplies data to only one layer
    // if it sends to several layers - it is safe to execute initial layer in any precision
    if (layer->outData[0]->getInputTo().size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->outData[0]->getInputTo().size() == 1) {" << std::endl;
        std::string aType = layer->outData[0]->getInputTo().begin()->second->type;
        if (CaselessEq<std::string>()(aType, "relu")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (CaselessEq<std::string>()(aType, 'relu')) {" << std::endl;
            return true;
        } else if (CaselessEq<std::string>()(aType, "clamp")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (CaselessEq<std::string>()(aType, 'clamp')) {" << std::endl;
            if (!isReLULikeClamp(layer->outData[0]->getInputTo().begin()->second)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (!isReLULikeClamp(layer->outData[0]->getInputTo().begin()->second)) {" << std::endl;
                return false;
            }
        } else {
            static const InferenceEngine::details::caseless_set<std::string> nonSuportedActivations = {
                "elu",  "clamp",  "tanh",        "logistic",  "square", "abs",
                "sqrt", "linear", "bounded_elu", "sort_relu", "relu6"};
            return nonSuportedActivations.find(aType) == nonSuportedActivations.end();
        }
    }
    return true;
}

bool CNNNetworkInt8Normalizer::isNextFusionAllowed(const CNNLayer::Ptr& layer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  bool CNNNetworkInt8Normalizer::isNextFusionAllowed(const CNNLayer::Ptr& layer) {" << std::endl;
    // fusion can happen only if initial layer supplies data to only one layer
    // if it sends to several layers - it is safe to execute initial layer in any precision
    if (layer->outData[0]->getInputTo().size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->outData[0]->getInputTo().size() == 1) {" << std::endl;
        std::string aType = layer->outData[0]->getInputTo().begin()->second->type;
        if (CaselessEq<std::string>()(aType, "relu")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (CaselessEq<std::string>()(aType, 'relu')) {" << std::endl;
            ReLULayer* rL = dynamic_cast<ReLULayer*>(layer->outData[0]->getInputTo().begin()->second.get());
            if (rL == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (rL == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "Layer " << layer->outData[0]->getInputTo().begin()->second->name
                                   << " is not instance of ReLULayer class";
            }
            if (rL->negative_slope != 0.f) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (rL->negative_slope != 0.f) {" << std::endl;
                return false;
            }
        } else if (CaselessEq<std::string>()(aType, "clamp")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (CaselessEq<std::string>()(aType, 'clamp')) {" << std::endl;
            if (!isReLULikeClamp(layer->outData[0]->getInputTo().begin()->second)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (!isReLULikeClamp(layer->outData[0]->getInputTo().begin()->second)) {" << std::endl;
                return false;
            }
        } else {
            static const InferenceEngine::details::caseless_set<std::string> nonSuportedActivations = {
                "elu",  "clamp",  "tanh",        "logistic",  "square", "abs",
                "sqrt", "linear", "bounded_elu", "sort_relu", "relu6"};
            return nonSuportedActivations.find(aType) == nonSuportedActivations.end();
        }
    } else {
        if (CaselessEq<std::string>()(layer->type, "eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (CaselessEq<std::string>()(layer->type, 'eltwise')) {" << std::endl;
            return false;
        }
    }
    return true;
}

bool CNNNetworkInt8Normalizer::isReLULikeClamp(CNNLayer::Ptr layer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  bool CNNNetworkInt8Normalizer::isReLULikeClamp(CNNLayer::Ptr layer) {" << std::endl;
    if (CaselessEq<std::string>()(layer->type, "Clamp")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (CaselessEq<std::string>()(layer->type, 'Clamp')) {" << std::endl;
        ClampLayer* clamp = dynamic_cast<ClampLayer*>(layer.get());
        if (clamp == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (clamp == nullptr) {" << std::endl;
            THROW_IE_EXCEPTION << "Int8 Normalizer error: cannot cast layer '" << layer->name << "' to Clamp";
        }
        return clamp->min_value == 0;
    }
    return false;
}

void CNNNetworkInt8Normalizer::DefinesExecutionPrecision(CNNNetwork& net, CNNStatisticHelper& statHelper) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  void CNNNetworkInt8Normalizer::DefinesExecutionPrecision(CNNNetwork& net, CNNStatisticHelper& statHelper) {" << std::endl;
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    // Converting layers to Int8. Calculating the multipliers if needed
    for (auto iter : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto iter : sortedLayers) {" << std::endl;
        if (iter->params.find("quantization_level") != iter->params.end() &&
            (iter->params["quantization_level"] == "FP32" || iter->params["quantization_level"] == "FP16")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              (iter->params['quantization_level'] == 'FP32' || iter->params['quantization_level'] == 'FP16')) {" << std::endl;
            continue;
        }

        // Legacy: FullyConnected should not be converted to Int8,
        // if it isn't explicitly marked to.
        if (iter->params.find("quantization_level") == iter->params.end() &&
            CaselessEq<std::string>()(iter->type, "fullyconnected")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              CaselessEq<std::string>()(iter->type, 'fullyconnected')) {" << std::endl;
            continue;
        }

        if (!statHelper.canLayerBeQuantized(iter)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (!statHelper.canLayerBeQuantized(iter)) {" << std::endl;
            continue;
        }

        if (CaselessEq<std::string>()(iter->type, "convolution") ||
            CaselessEq<std::string>()(iter->type, "fullyconnected")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              CaselessEq<std::string>()(iter->type, 'fullyconnected')) {" << std::endl;
            if (canLayerBeI8(iter)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (canLayerBeI8(iter)) {" << std::endl;
                iter->precision = Precision::I8;
                // we will override I8 to U8 during analysing of Conv-ReLU and Conv-Sum-ReLU fusions
                iter->outData[0]->setPrecision(Precision::I8);
            }
        } else if (CaselessEq<std::string>()(iter->type, "relu") || isReLULikeClamp(iter)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (CaselessEq<std::string>()(iter->type, 'relu') || isReLULikeClamp(iter)) {" << std::endl;
            // casting to ReLU
            ReLULayer* rL = dynamic_cast<ReLULayer*>(iter.get());
            DataPtr outData = iter->outData.size() ? iter->outData[0] : nullptr;
            auto inputData = iter->insData[0].lock();
            if (inputData && inputData->getCreatorLayer().lock()->precision != Precision::FP32 &&
                outData->getPrecision() == Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  outData->getPrecision() == Precision::FP32) {" << std::endl;
                iter->precision = Precision::I8;
                if (rL != nullptr && rL->negative_slope != 0.0f) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (rL != nullptr && rL->negative_slope != 0.0f) {" << std::endl;
                    outData->setPrecision(Precision::I8);
                } else {
                    outData->setPrecision(Precision::U8);
                    // if convolution is a predecessor, change its data to U8 also
                    CNNLayer::Ptr prevLayer = inputData->getCreatorLayer().lock();
                    if (prevLayer && (CaselessEq<std::string>()(prevLayer->type, "convolution") ||
                                      CaselessEq<std::string>()(prevLayer->type, "fullyconnected") ||
                                      CaselessEq<std::string>()(prevLayer->type, "eltwise"))) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                        CaselessEq<std::string>()(prevLayer->type, 'eltwise'))) {" << std::endl;
                        if (!isNextFusionAllowed(prevLayer) && inputData->getPrecision() == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          if (!isNextFusionAllowed(prevLayer) && inputData->getPrecision() == Precision::I8) {" << std::endl;
                            outData->setPrecision(Precision::I8);
                        } else {
                            inputData->setPrecision(Precision::U8);
                        }
                    }
                    // if there is a patter A0 -> Eltwise -> ReLU and Convolution -> Eltwise -> ReLU,
                    // need to mark data after conv as U8
                    if (prevLayer && CaselessEq<std::string>()(prevLayer->type, "eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (prevLayer && CaselessEq<std::string>()(prevLayer->type, 'eltwise')) {" << std::endl;
                        // decising which input will be used for fusion conv-sum-relu
                        CNNLayer::Ptr input1 = prevLayer->insData[0].lock()->getCreatorLayer().lock();
                        CNNLayer::Ptr input2 = prevLayer->insData[1].lock()->getCreatorLayer().lock();
                        CNNLayer::Ptr convLayer = nullptr;
                        CNNLayer::Ptr sumLayer = nullptr;

                        if (!CaselessEq<std::string>()(input1->type, "convolution")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          if (!CaselessEq<std::string>()(input1->type, 'convolution')) {" << std::endl;
                            sumLayer = input1;
                            convLayer = input2;
                        } else {
                            // it covers a case when both inputs are convolutions or when first input is not convolution
                            convLayer = input1;
                            sumLayer = input2;
                        }
                        convLayer->outData[0]->setPrecision(sumLayer->outData[0]->getPrecision());
                    }
                }
            }
        } else if (CaselessEq<std::string>()(iter->type, "pooling")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (CaselessEq<std::string>()(iter->type, 'pooling')) {" << std::endl;
            auto pool = dynamic_cast<PoolingLayer*>(iter.get());
            if (pool == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (pool == nullptr) {" << std::endl;
                THROW_IE_EXCEPTION << "Int8 Normalizer error: cannot cast layer '" << iter->name << "' to pooling";
            }

            if (pool->_type == PoolingLayer::MAX || (pool->_type == PoolingLayer::AVG && pool->outData.size() == 1)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (pool->_type == PoolingLayer::MAX || (pool->_type == PoolingLayer::AVG && pool->outData.size() == 1)) {" << std::endl;
                auto prevLayer = iter->insData[0].lock()->getCreatorLayer().lock();
                if (prevLayer && (prevLayer->precision == Precision::I8 || prevLayer->precision == Precision::U8)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (prevLayer && (prevLayer->precision == Precision::I8 || prevLayer->precision == Precision::U8)) {" << std::endl;
                    iter->precision = Precision::I8;
                    iter->outData[0]->setPrecision(statHelper.hasNegativeOutput(iter->name) ? Precision::I8
                                                                                            : Precision::U8);
                }
            }
        } else if (CaselessEq<std::string>()(iter->type, "concat")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (CaselessEq<std::string>()(iter->type, 'concat')) {" << std::endl;
            // we can do safe
            // casting to concat and take axis parameter
            // we can concat scales only if concat does concatination by feature maps
            bool axisFeatureMaps = false;
            auto concatLayer = dynamic_cast<ConcatLayer*>(iter.get());
            if (concatLayer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (concatLayer) {" << std::endl;
                if (concatLayer->_axis == 1 && concatLayer->insData.size() &&
                    concatLayer->insData[0].lock()->getTensorDesc().getDims().size() == 4) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      concatLayer->insData[0].lock()->getTensorDesc().getDims().size() == 4) {" << std::endl;
                    axisFeatureMaps = true;
                }
            } else {
                THROW_IE_EXCEPTION << "Int8 Normalizer error: cannot cast layer " << iter->name << " to concat";
            }

            if (axisFeatureMaps) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (axisFeatureMaps) {" << std::endl;
                // verification of input data types
                bool inputFP32 = false;
                bool inputI8 = false;
                bool inputU8 = false;

                for (auto inputData : iter->insData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (auto inputData : iter->insData) {" << std::endl;
                    auto data = inputData.lock();
                    if (data->getPrecision() == Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (data->getPrecision() == Precision::FP32) {" << std::endl;
                        inputFP32 = true;
                    } else if (data->getPrecision() == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      } else if (data->getPrecision() == Precision::I8) {" << std::endl;
                        inputI8 = true;
                    } else if (data->getPrecision() == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      } else if (data->getPrecision() == Precision::U8) {" << std::endl;
                        inputU8 = true;
                    } else {
                        // Is it a case of input, i.e. passing I16 to concat?
                        // TODO(amalyshe) to handle inputs as a separate usecase
                        THROW_IE_EXCEPTION << "I8 normalizer: input data has unknown precision on the edge for concat: "
                                           << data->getName();
                    }
                }

                if (inputFP32) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (inputFP32) {" << std::endl;
                    for (auto i : iter->insData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      for (auto i : iter->insData) {" << std::endl;
                        if (i.lock()->getCreatorLayer().lock()->precision != Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          if (i.lock()->getCreatorLayer().lock()->precision != Precision::FP32) {" << std::endl;
                            returnTailToFP32(i.lock()->getCreatorLayer().lock());
                        }
                    }
                } else {
                    iter->precision = Precision::I8;

                    // we set outpout precision to U8 only if all inputs are U8, in other case it will be I8
                    auto outputPrecision = (inputU8 && !inputI8) ? Precision::U8 : Precision::I8;

                    // if we have mixed input for I8 and U8, we have to insert scale to edges having U8 to convert to I8
                    // Yes, it leads to loosing of some precision and might lead to some performance degradation
                    // until we have scale supporting s8/u8 input and s8/u8 output.
                    if (inputU8 && inputI8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (inputU8 && inputI8) {" << std::endl;
                        // looking for all edges having U8
                        for (size_t d = 0; d < iter->insData.size(); d++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          for (size_t d = 0; d < iter->insData.size(); d++) {" << std::endl;
                            auto data = iter->insData[d].lock();
                            if (data->getPrecision() == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                              if (data->getPrecision() == Precision::U8) {" << std::endl;
                                const size_t c = static_cast<size_t>(data->getDims()[1]);
                                std::vector<float> ssWValues(c, 1.0f);
                                std::vector<float> ssSValues(c, 0.0f);

                                std::string layerName =
                                    data->getCreatorLayer().lock()->name + "_Concat_ScaleShift_U8I8_" + iter->name;
                                CNNLayer::Ptr newLayer =
                                    createDWConvolutionForScale(layerName, c, ssWValues.data(), ssSValues.data());
                                newLayer->precision = Precision::I8;
                                AddLayerToCNNNetworkBeforeLayer(newLayer, iter, d);

                                // update statistic to pass quantization smoothly
                                std::string inputLayerName =
                                    newLayer->insData[0].lock()->getCreatorLayer().lock()->name;
                                statHelper.copyStatistics(inputLayerName, layerName);
                                newLayer->outData[0]->setPrecision(Precision::I8);
                            }
                        }
                    }

                    if (iter->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (iter->outData.size() == 1) {" << std::endl;
                        for (auto&& out : iter->outData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          for (auto&& out : iter->outData) {" << std::endl;
                            out->setPrecision(outputPrecision);
                        }
                    }
                }
            }
        } else if (CaselessEq<std::string>()(iter->type, "eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (CaselessEq<std::string>()(iter->type, 'eltwise')) {" << std::endl;
            // we decide which of the layers will be in int-8 mode and initialize special scale which will be used
            // later in "conv-sum-relu" fuse. i8 execution of eltwise always assume this fusion
            if (canLayerBeI8(iter)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (canLayerBeI8(iter)) {" << std::endl;
                if (iter->insData.size() == 2) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (iter->insData.size() == 2) {" << std::endl;
                    CNNLayer::Ptr input1 = iter->insData[0].lock()->getCreatorLayer().lock();
                    CNNLayer::Ptr input2 = iter->insData[1].lock()->getCreatorLayer().lock();
                    if ((CaselessEq<std::string>()(input1->type, "convolution") ||
                         CaselessEq<std::string>()(input2->type, "convolution")) &&
                        !CaselessEq<std::string>()(input1->type, "concat") &&
                        !CaselessEq<std::string>()(input2->type, "concat") && input1->precision != Precision::FP32 &&
                        input2->precision != Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          input2->precision != Precision::FP32) {" << std::endl;
                        // understand which layer will be used for sum
                        CNNLayer::Ptr sumLayer = nullptr;
                        CNNLayer::Ptr convLayer = nullptr;

                        if (!CaselessEq<std::string>()(input1->type, "convolution")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          if (!CaselessEq<std::string>()(input1->type, 'convolution')) {" << std::endl;
                            sumLayer = input1;
                            convLayer = input2;
                        } else {
                            // it covers a case when both inputs are convolutions or when first input is not convolution
                            sumLayer = input2;
                            convLayer = input1;
                        }

                        // if we find supported activation, mark it's output as I8 or U8 depending on statistics
                        if (iter->outData.size() == 1 && iter->outData[0]->getInputTo().size() == 1 &&
                            (CaselessEq<std::string>()(iter->outData[0]->getInputTo().begin()->second->type, "ReLU") ||
                             CNNNetworkInt8Normalizer::isReLULikeClamp(
                                 iter->outData[0]->getInputTo().begin()->second))) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                   iter->outData[0]->getInputTo().begin()->second))) {" << std::endl;
                            auto activation = iter->outData[0]->getInputTo().begin()->second;
                            activation->precision = Precision::I8;
                            if (!statHelper.hasNegativeOutput(statHelper.getLatestInFuse(convLayer)->name)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                              if (!statHelper.hasNegativeOutput(statHelper.getLatestInFuse(convLayer)->name)) {" << std::endl;
                                activation->outData[0]->setPrecision(Precision::U8);
                                iter->outData[0]->setPrecision(Precision::U8);
                            } else {
                                activation->outData[0]->setPrecision(Precision::I8);
                                iter->outData[0]->setPrecision(Precision::I8);
                            }
                        } else {
                            iter->outData[0]->setPrecision(Precision::I8);
                        }

                        if (convLayer->outData[0]->getTensorDesc().getPrecision() == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          if (convLayer->outData[0]->getTensorDesc().getPrecision() == Precision::I8) {" << std::endl;
                            // verify precision on input edges before and after eltwise fusion
                            // if we have i8/u8 missmatch between sum layer input and conv-sum-activation output,
                            // then in this case we have to add requantization to i8 on sum input edge
                            auto latestInFuse = statHelper.getLatestInFuse(convLayer);
                            if (latestInFuse->outData[0]->getTensorDesc().getPrecision() == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                              if (latestInFuse->outData[0]->getTensorDesc().getPrecision() == Precision::I8) {" << std::endl;
                                if (input1 == sumLayer &&
                                    iter->insData[0].lock()->getTensorDesc().getPrecision() == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                      iter->insData[0].lock()->getTensorDesc().getPrecision() == Precision::U8) {" << std::endl;
                                    sumLayer = addU8ToI8Conversion(iter->insData[0].lock(), iter, statHelper);
                                } else if (input2 == sumLayer &&
                                           iter->insData[1].lock()->getTensorDesc().getPrecision() == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                             iter->insData[1].lock()->getTensorDesc().getPrecision() == Precision::U8) {" << std::endl;
                                    sumLayer = addU8ToI8Conversion(iter->insData[0].lock(), iter, statHelper);
                                }
                                if (!sumLayer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                  if (!sumLayer) {" << std::endl;
                                    THROW_IE_EXCEPTION << "I8 normalizer had to add U8->I8 conversion before "
                                                       << iter->name << " but failed to do this";
                                }
                            }

                            // mark eltwise as a I8 executable, mark out data as I8
                            iter->precision = Precision::I8;
                            convLayer->outData[0]->setPrecision(sumLayer->outData[0]->getPrecision());
                            // calculate the only scale
                            Blob::Ptr sumLayerScales = statHelper.getOutputScale(statHelper.getLatestInFuse(sumLayer));
                            Blob::Ptr convLayerScales =
                                statHelper.getOutputScale(statHelper.getLatestInFuse(convLayer));
                            float* sumScale = sumLayerScales->buffer().as<float*>();
                            float* convScale = convLayerScales->buffer().as<float*>();
                            for (size_t i = 0; i < sumLayerScales->size(); i++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                              for (size_t i = 0; i < sumLayerScales->size(); i++) {" << std::endl;
                                sumScale[i] /= convScale[i];
                            }

                            iter->blobs["eltwise-sum-scale"] = sumLayerScales;
                        }
                    }
                }
            } else {
                // if there are convolutions are inputs to this eltwise, we forcedly move them to FP32
                for (auto i : iter->insData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (auto i : iter->insData) {" << std::endl;
                    auto type = i.lock()->getCreatorLayer().lock()->type;
                    if (CaselessEq<std::string>()(type, "convolution") ||
                        CaselessEq<std::string>()(type, "fullyconnected")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          CaselessEq<std::string>()(type, 'fullyconnected')) {" << std::endl;
                        i.lock()->getCreatorLayer().lock()->precision = Precision::FP32;
                        i.lock()->setPrecision(Precision::FP32);
                    }
                }
            }
        } else if (CaselessEq<std::string>()(iter->type, "resample")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          } else if (CaselessEq<std::string>()(iter->type, 'resample')) {" << std::endl;
            iter->precision = Precision::I8;
            iter->outData[0]->setPrecision(iter->insData[0].lock()->getPrecision());
        }
    }

    // quantization of weights/biases
    sortedLayers = CNNNetSortTopologically(net);
    for (auto iter : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto iter : sortedLayers) {" << std::endl;
        if (iter->precision == Precision::I8 && (CaselessEq<std::string>()(iter->type, "convolution") ||
                                                 CaselessEq<std::string>()(iter->type, "fullyconnected"))) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                                   CaselessEq<std::string>()(iter->type, 'fullyconnected'))) {" << std::endl;
            QuantizeConvolutionOrFullyConnected(iter, statHelper);
        }
    }

    // Returning of tails to FP32 mode if optimistic approach marked them as I8
    // no sense to do pooling in i8, we can return just after convolution
    for (auto iter : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto iter : sortedLayers) {" << std::endl;
        // TODO(amalyshe) here is a handling of case when iter provides data to the only one next layer
        // need to extend to cases when it provides data to many layers
        if (iter->precision == Precision::I8 && iter->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (iter->precision == Precision::I8 && iter->outData.size() == 1) {" << std::endl;
            if ((iter->outData[0]->getInputTo().size() == 1 &&
                 iter->outData[0]->getInputTo().begin()->second->precision == Precision::FP32) ||
                iter->outData[0]->getInputTo().size() == 0) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  iter->outData[0]->getInputTo().size() == 0) {" << std::endl;
                returnTailToFP32(iter);
            }
        }
    }
}

void CNNNetworkInt8Normalizer::PropagateScaleFactors(CNNNetwork& net, const CNNStatisticHelper& statHelper) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  void CNNNetworkInt8Normalizer::PropagateScaleFactors(CNNNetwork& net, const CNNStatisticHelper& statHelper) {" << std::endl;
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    // Moving o-scales down
    for (auto iter : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto iter : sortedLayers) {" << std::endl;
        if (iter->type == "Concat" && iter->precision == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (iter->type == 'Concat' && iter->precision == Precision::I8) {" << std::endl;
            // Checking if all inputs are INT8
            bool all_inputs_are_int8 = true;
            for (int k = 0; k < iter->insData.size(); k++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (int k = 0; k < iter->insData.size(); k++) {" << std::endl;
                auto prevKLayer = iter->insData[k].lock()->getCreatorLayer().lock();
                if ((prevKLayer->precision != Precision::I8 && prevKLayer->precision != Precision::U8) ||
                    prevKLayer->blobs.find("i-concat-scale") == prevKLayer->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      prevKLayer->blobs.find('i-concat-scale') == prevKLayer->blobs.end()) {" << std::endl;
                    all_inputs_are_int8 = false;
                    break;
                }
            }

            if (all_inputs_are_int8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (all_inputs_are_int8) {" << std::endl;
                // Merging o-scales of the inputs to make one for the Concat
                // Creating the o-scale for the Concat by concatenating the input concats
                size_t outputChannels = iter->outData[0]->getTensorDesc().getDims()[1];

                std::shared_ptr<Data> oScaleData =
                    std::shared_ptr<Data>(new Data("o-scale", {Precision::FP32, {outputChannels}, Layout::C}));
                auto oScale = CreateBlobFromData(oScaleData);
                oScale->allocate();

                float* oScaleMemory = static_cast<float*>(oScale->buffer());
                int cc = 0;
                for (int in = 0; in < iter->insData.size(); in++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (int in = 0; in < iter->insData.size(); in++) {" << std::endl;
                    auto prevOScale = iter->insData[in].lock()->getCreatorLayer().lock()->blobs["i-concat-scale"];
                    float* prevOScaleMemory = static_cast<float*>(prevOScale->buffer());

                    for (int c = 0; c < prevOScale->size(); c++) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      for (int c = 0; c < prevOScale->size(); c++) {" << std::endl;
                        oScaleMemory[cc] = prevOScaleMemory[c];
                        cc++;
                    }
                }
                if (cc != outputChannels)
                    THROW_IE_EXCEPTION << "Size of o-scale after " << iter->name
                                       << " isn't equal to the channels count";

                iter->precision = Precision::I8;
                iter->blobs["o-scale"] = oScale;
            }
        }

        if (iter->blobs.find("o-scale") != iter->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (iter->blobs.find('o-scale') != iter->blobs.end()) {" << std::endl;
            int int8Consumers = 0;
            int fp32Consumers = 0;
            if (iter->outData.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (iter->outData.size() > 1) {" << std::endl;
                THROW_IE_EXCEPTION << "normalization algorithm for int8 found layer having o-scale and multiple ports";
            }
            if (iter->outData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (iter->outData.size() == 1) {" << std::endl;
                for (auto l : iter->outData[0]->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (auto l : iter->outData[0]->getInputTo()) {" << std::endl;
                    if (l.second->precision == Precision::I8 || l.second->precision == Precision::U8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (l.second->precision == Precision::I8 || l.second->precision == Precision::U8) {" << std::endl;
                        if (CaselessEq<std::string>()(l.second->type, "Pooling") ||
                            CaselessEq<std::string>()(l.second->type, "ReLU") ||
                            CNNNetworkInt8Normalizer::isReLULikeClamp(l.second)) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                              CNNNetworkInt8Normalizer::isReLULikeClamp(l.second)) {" << std::endl;
                            l.second->blobs["o-scale"] = iter->blobs["o-scale"];
                            // debug scales. Need to compare with actual values in FP32 scoring
                            l.second->blobs["ext-scale"] = l.second->blobs["o-scale"];
                            int8Consumers++;
                        } else if (l.second->type == "Convolution") {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          } else if (l.second->type == 'Convolution') {" << std::endl;
                            l.second->blobs.erase("i-scale");
                            int8Consumers++;
                        } else if (CaselessEq<std::string>()(l.second->type, "Eltwise")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          } else if (CaselessEq<std::string>()(l.second->type, 'Eltwise')) {" << std::endl;
                            if (statHelper.getLatestInFuse(iter) != iter) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                              if (statHelper.getLatestInFuse(iter) != iter) {" << std::endl;
                                l.second->blobs["o-scale"] = iter->blobs["o-scale"];
                            }
                            int8Consumers++;
                        } else if ((l.second->precision == Precision::I8 || l.second->precision == Precision::U8) &&
                                   CaselessEq<std::string>()(l.second->type, "Resample")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                     CaselessEq<std::string>()(l.second->type, 'Resample')) {" << std::endl;
                            // If resample has concat as input layer it should inherit it's
                            // output scale
                            if (l.second->insData.size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                              if (l.second->insData.size() == 1) {" << std::endl;
                                CNNLayerPtr creator = l.second->insData[0].lock()->getCreatorLayer().lock();
                                if (CaselessEq<std::string>()(creator->type, "Concat")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                  if (CaselessEq<std::string>()(creator->type, 'Concat')) {" << std::endl;
                                    l.second->blobs["o-scale"] = creator->blobs["o-scale"];
                                    l.second->blobs["i-concat-scale"] = l.second->blobs["o-scale"];
                                }
                            }

                            // No concat found, let use statistics
                            if (l.second->blobs.find("o-scale") == l.second->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                              if (l.second->blobs.find('o-scale') == l.second->blobs.end()) {" << std::endl;
                                auto oScale = statHelper.getOutputScale(l.second);
                                l.second->blobs["o-scale"] = oScale;
                                l.second->blobs["i-concat-scale"] = l.second->blobs["o-scale"];
                            }
                            int8Consumers++;
                        } else if ((l.second->precision == Precision::I8) &&
                                   CaselessEq<std::string>()(l.second->type, "concat")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                     CaselessEq<std::string>()(l.second->type, 'concat')) {" << std::endl;
                            // if concat is i8, we can propagate oscale further to concat.
                            // The logic around o-scale assumes that if we have it in the layer after iteration
                            // in this loop it means that it must not be removed and we need to place
                            // scale. While for concat we return to one layer back and again need to analyze o-scale
                            // and it is not clear if we need to return o-scale or it was only for concat.
                            // Having all of this in mind, it's better to rename o-scale to i-concat-scale
                            iter->blobs["i-concat-scale"] = iter->blobs["o-scale"];
                            int8Consumers++;
                        } else {
                            fp32Consumers++;
                        }
                    } else if (CaselessEq<std::string>()(l.second->type, "priorbox") ||
                               CaselessEq<std::string>()(l.second->type, "priorboxclustered")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                                 CaselessEq<std::string>()(l.second->type, 'priorboxclustered')) {" << std::endl;
                    } else {
                        // we are leaving o-scale still for adding of scale-shift before FP32 layer
                        fp32Consumers++;
                    }
                }

                if (iter->outData[0]->getInputTo().empty()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (iter->outData[0]->getInputTo().empty()) {" << std::endl;
                    fp32Consumers++;
                }

                if (CaselessEq<std::string>()(iter->type, "Convolution") ||
                    CaselessEq<std::string>()(iter->type, "FullyConnected")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      CaselessEq<std::string>()(iter->type, 'FullyConnected')) {" << std::endl;
                    if (int8Consumers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (int8Consumers) {" << std::endl;
                        iter->blobs["oi-scale"] = iter->blobs["o-scale"];
                    } else {
                        iter->outData[0]->setPrecision(Precision::FP32);
                    }
                }
                if (!fp32Consumers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  if (!fp32Consumers) {" << std::endl;
                    iter->blobs.erase("o-scale");
                }
            }
        }
    }

    // fixing cornercases when o-scale was propagated through linear tail but it is more efficient to leave
    // conversion to de-normalized values in convolution
    for (auto iter : sortedLayers) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto iter : sortedLayers) {" << std::endl;
        if (iter->blobs.find("o-scale") != iter->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:          if (iter->blobs.find('o-scale') != iter->blobs.end()) {" << std::endl;
            // go over out data. if all outputs are fp32, continue this optimization
            bool canOptimize = true;

            // current layer must not be convolution
            if (CaselessEq<std::string>()(iter->type, "convolution")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (CaselessEq<std::string>()(iter->type, 'convolution')) {" << std::endl;
                canOptimize = false;
            }
            for (auto o : iter->outData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              for (auto o : iter->outData) {" << std::endl;
                for (auto ol : o->getInputTo()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (auto ol : o->getInputTo()) {" << std::endl;
                    if (ol.second->precision == Precision::I8) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (ol.second->precision == Precision::I8) {" << std::endl;
                        canOptimize = false;
                    }
                }
            }
            if (!canOptimize) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (!canOptimize) {" << std::endl;
                continue;
            }
            // trying to go up until convolution
            auto curLayer = iter;
            bool eliminateOScale = true;
            while (curLayer && curLayer->blobs.find("oi-scale") == curLayer->blobs.end() && eliminateOScale) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              while (curLayer && curLayer->blobs.find('oi-scale') == curLayer->blobs.end() && eliminateOScale) {" << std::endl;
                if (curLayer->insData.size() == 1 && curLayer->insData[0].lock()->getCreatorLayer().lock() &&
                    curLayer->insData[0].lock()->getCreatorLayer().lock()->outData.size() == 1 &&
                    curLayer->insData[0].lock()->getInputTo().size() == 1) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      curLayer->insData[0].lock()->getInputTo().size() == 1) {" << std::endl;
                    curLayer = curLayer->insData[0].lock()->getCreatorLayer().lock();
                    if (!CaselessEq<std::string>()(curLayer->type, "Pooling") &&
                        !CaselessEq<std::string>()(curLayer->type, "ReLU") && !isReLULikeClamp(curLayer) &&
                        !CaselessEq<std::string>()(curLayer->type, "Convolution")) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                          !CaselessEq<std::string>()(curLayer->type, 'Convolution')) {" << std::endl;
                        eliminateOScale = false;
                    }
                } else {
                    eliminateOScale = false;
                }
            }
            if (eliminateOScale && curLayer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:              if (eliminateOScale && curLayer) {" << std::endl;
                for (auto o : iter->outData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (auto o : iter->outData) {" << std::endl;
                    o->setPrecision(Precision::FP32);
                }
                for (auto o : curLayer->outData) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  for (auto o : curLayer->outData) {" << std::endl;
                    o->setPrecision(Precision::FP32);
                }

                curLayer->blobs.erase("oi-scale");
                iter->blobs.erase("o-scale");
                auto iLayer = iter;
                while (iLayer != curLayer) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                  while (iLayer != curLayer) {" << std::endl;
                    if (iLayer->type == "Pooling") {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                      if (iLayer->type == 'Pooling') {" << std::endl;
                        iLayer->precision = Precision::FP32;
                    }
                    iLayer = iLayer->insData[0].lock()->getCreatorLayer().lock();
                }
            }
        }
    }
}

std::string getBlobDimention(const Blob::Ptr blob) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  std::string getBlobDimention(const Blob::Ptr blob) {" << std::endl;
    size_t idx = blob->getTensorDesc().getDims().size();

    std::stringstream blobDimention;
    blobDimention << "[";
    for (auto& dim : blob->getTensorDesc().getDims()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      for (auto& dim : blob->getTensorDesc().getDims()) {" << std::endl;
        blobDimention << dim << ((--idx) != 0u ? ", " : "");
    }
    blobDimention << "]";

    return blobDimention.str();
}

void precisionColoring(const CNNLayerPtr layer, ordered_properties& printed_properties,
                       ordered_properties& node_properties) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:                         ordered_properties& node_properties) {" << std::endl;
    // looking for the w-scale
    if (layer->blobs.find("w-scale") != layer->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->blobs.find('w-scale') != layer->blobs.end()) {" << std::endl;
        printed_properties.insert(
            printed_properties.begin(),
            std::pair<std::string, std::string>("w-scale", getBlobDimention(layer->blobs.find("w-scale")->second)));
    }

    // looking for the oi-scale
    if (layer->blobs.find("oi-scale") != layer->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->blobs.find('oi-scale') != layer->blobs.end()) {" << std::endl;
        printed_properties.insert(
            printed_properties.begin(),
            std::pair<std::string, std::string>("oi-scale", getBlobDimention(layer->blobs.find("oi-scale")->second)));
    }

    // looking for the o-scale
    if (layer->blobs.find("o-scale") != layer->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->blobs.find('o-scale') != layer->blobs.end()) {" << std::endl;
        printed_properties.insert(
            printed_properties.begin(),
            std::pair<std::string, std::string>("o-scale", getBlobDimention(layer->blobs.find("o-scale")->second)));
    }
    // looking for the i-scale
    if (layer->blobs.find("i-scale") != layer->blobs.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->blobs.find('i-scale') != layer->blobs.end()) {" << std::endl;
        printed_properties.insert(
            printed_properties.begin(),
            std::pair<std::string, std::string>("i-scale", getBlobDimention(layer->blobs.find("i-scale")->second)));
    }

    printed_properties.insert(
        printed_properties.begin(),
        std::pair<std::string, std::string>("Precision", layer->precision == Precision::FP32 ? "FP32" : "I8"));

    if (layer->precision == Precision::FP32) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      if (layer->precision == Precision::FP32) {" << std::endl;
        node_properties.emplace_back("fillcolor", "#5A5DF0");
    } else {
        node_properties.emplace_back("fillcolor", "#20F608");
    }
}

void CNNNetworkInt8Normalizer::NormalizeNetwork(ICNNNetwork& network, ICNNNetworkStats& netStats) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:  void CNNNetworkInt8Normalizer::NormalizeNetwork(ICNNNetwork& network, ICNNNetworkStats& netStats) {" << std::endl;
    CNNNetwork cnnn(ICNNNetwork::Ptr(&network, [](void*) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_int8_normalizer.cpp:      CNNNetwork cnnn(ICNNNetwork::Ptr(&network, [](void*) {" << std::endl;}));

    int maxSign = 0x7F;
    int maxUnsign = 0xFF;

    // Applying int8-conversion
    StatsMap statsMap = netStats.getNodesStats();

    CNNStatisticHelper statHelper(cnnn, statsMap, maxSign, maxUnsign);

    replaceScaleShiftByDWConvolution(cnnn);

    DefinesExecutionPrecision(cnnn, statHelper);
    PropagateScaleFactors(cnnn, statHelper);
    ClampsToReLU(cnnn, statHelper);
    AddScaleShifts(cnnn, statHelper);
#ifndef NDEBUG
    std::ofstream file("i8_normalized.dot");
    saveGraphToDot(cnnn, file, precisionColoring);
#endif
}
