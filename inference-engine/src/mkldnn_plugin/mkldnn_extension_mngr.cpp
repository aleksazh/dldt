#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <w_unistd.h>
#include <w_dirent.h>
#include <debug.h>
#include <algorithm>
#include <file_utils.h>

#include "mkldnn_extension_mngr.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

void MKLDNNExtensionManager::AddExtension(IExtensionPtr extension) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp:  void MKLDNNExtensionManager::AddExtension(IExtensionPtr extension) {" << std::endl;
    _extensions.push_back(extension);
}

InferenceEngine::ILayerImplFactory* MKLDNNExtensionManager::CreateExtensionFactory(
        const InferenceEngine::CNNLayerPtr &layer) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp:          const InferenceEngine::CNNLayerPtr &layer) {" << std::endl;
    if (!layer)
        THROW_IE_EXCEPTION << "Cannot get cnn layer!";
    ILayerImplFactory* factory = nullptr;
    for (auto& ext : _extensions) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp:      for (auto& ext : _extensions) {" << std::endl;
        ResponseDesc responseDesc;
        StatusCode rc;
        rc = ext->getFactoryFor(factory, layer.get(), &responseDesc);
        if (rc != OK) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp:          if (rc != OK) {" << std::endl;
            factory = nullptr;
            continue;
        }
        if (factory != nullptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp:          if (factory != nullptr) {" << std::endl;
            break;
        }
    }
    return factory;
}

IShapeInferImpl::Ptr MKLDNNExtensionManager::CreateReshaper(const InferenceEngine::CNNLayerPtr &layer) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp:  IShapeInferImpl::Ptr MKLDNNExtensionManager::CreateReshaper(const InferenceEngine::CNNLayerPtr &layer) {" << std::endl;
    if (!layer)
        THROW_IE_EXCEPTION << "Cannot get cnn layer!";
    IShapeInferImpl::Ptr reshaper = nullptr;
    for (auto& ext : _extensions) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp:      for (auto& ext : _extensions) {" << std::endl;
        ResponseDesc responseDesc;
        StatusCode rc;
        rc = ext->getShapeInferImpl(reshaper, layer->type.c_str(), &responseDesc);
        if (rc != OK) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp:          if (rc != OK) {" << std::endl;
            reshaper = nullptr;
            continue;
        }
        if (reshaper != nullptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_extension_mngr.cpp:          if (reshaper != nullptr) {" << std::endl;
            break;
        }
    }
    return reshaper;
}



