#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_stats_impl.hpp"

#include <ie_common.h>

#include <cassert>
#include <cfloat>
#include <fstream>
#include <map>
#include <memory>
#include <pugixml.hpp>
#include <string>
#include <vector>

#include "debug.h"

using namespace std;
namespace InferenceEngine {
namespace details {

CNNNetworkStatsImpl::~CNNNetworkStatsImpl() {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_stats_impl.cpp:  CNNNetworkStatsImpl::~CNNNetworkStatsImpl() {" << std::endl;}

void CNNNetworkStatsImpl::setNodesStats(const NetworkStatsMap& stats) {
    std::cerr << "./inference-engine/src/inference_engine/cnn_network_stats_impl.cpp:  void CNNNetworkStatsImpl::setNodesStats(const NetworkStatsMap& stats) {" << std::endl;
    netNodesStats = stats;
}

const NetworkStatsMap& CNNNetworkStatsImpl::getNodesStats() const {
    return netNodesStats;
}

}  // namespace details
}  // namespace InferenceEngine
