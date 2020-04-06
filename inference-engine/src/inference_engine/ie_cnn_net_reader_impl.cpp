#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <ie_cnn_net_reader_impl.h>

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cnn_network_ngraph_impl.hpp"
#include "debug.h"
#include "details/os/os_filesystem.hpp"
#include "ie_format_parser.h"
#include "ie_ir_reader.hpp"
#include "ie_profiling.hpp"
#include "parsers.h"
#include "xml_parse_utils.h"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

IE_SUPPRESS_DEPRECATED_START
CNNNetReaderImpl::CNNNetReaderImpl(const FormatParserCreator::Ptr& _creator)
    : parseSuccess(false), _version(0), parserCreator(_creator) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      : parseSuccess(false), _version(0), parserCreator(_creator) {" << std::endl;}

StatusCode CNNNetReaderImpl::SetWeights(const TBlob<uint8_t>::Ptr& weights, ResponseDesc* desc) noexcept {
    if (!_parser && _version < 10) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      if (!_parser && _version < 10) {" << std::endl;
        return DescriptionBuffer(desc) << "network must be read first";
    }
    try {
        if (_version == 10) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:          if (_version == 10) {" << std::endl;
#if defined(ENABLE_IR_READER) && defined(ENABLE_NGRAPH)
            // It's time to perform actual reading of V10 network and instantiate CNNNetworkNGraphImpl
            IRReader v10Reader(extensions);
            std::stringstream model;
            xmlDoc->save(model);
            network = std::make_shared<CNNNetworkNGraphImpl>(v10Reader.read(model.str(), weights));
#else
            return DescriptionBuffer(desc) << "Please, recompile Inference Engine with the ENABLE_IR_READER=ON AND "
                                              "ENABLE_NGRAPH=ON Cmake option";
#endif
        } else {
            _parser->SetWeights(weights);
        }
    } catch (const InferenceEngineException& iee) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      } catch (const InferenceEngineException& iee) {" << std::endl;
        xmlDoc.reset();
        return DescriptionBuffer(desc) << iee.what();
    }

    xmlDoc.reset();
    return OK;
}

size_t CNNNetReaderImpl::GetFileVersion(pugi::xml_node& root) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:  size_t CNNNetReaderImpl::GetFileVersion(pugi::xml_node& root) {" << std::endl;
    return XMLParseUtils::GetUIntAttr(root, "version", 0);
}

StatusCode CNNNetReaderImpl::ReadNetwork(const void* model, size_t size, ResponseDesc* resp) noexcept {
    if (network) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      if (network) {" << std::endl;
        return DescriptionBuffer(NETWORK_NOT_READ, resp)
               << "Network has been read already, use new reader instance to read new network.";
    }

    xmlDoc = std::make_shared<pugi::xml_document>();
    pugi::xml_parse_result res = xmlDoc->load_buffer(model, size);
    if (res.status != pugi::status_ok) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      if (res.status != pugi::status_ok) {" << std::endl;
        return DescriptionBuffer(resp) << res.description() << "at offset " << res.offset;
    }
    StatusCode ret = ReadNetwork();
    if (ret != OK) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      if (ret != OK) {" << std::endl;
        return DescriptionBuffer(resp) << "Error reading network: " << description;
    }
    return OK;
}

StatusCode CNNNetReaderImpl::ReadWeights(const char* filepath, ResponseDesc* resp) noexcept {
    IE_PROFILING_AUTO_SCOPE(CNNNetReaderImpl::ReadWeights)
    int64_t fileSize = FileUtils::fileSize(filepath);

    if (fileSize < 0)
        return DescriptionBuffer(resp) << "filesize for: " << filepath << " - " << fileSize
                                       << "<0. Please, check weights file existence.";

    // If IR V10 then there hasn't been loaded network yet
    if (network.get() == nullptr && _version < 10) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      if (network.get() == nullptr && _version < 10) {" << std::endl;
        return DescriptionBuffer(resp) << "network is empty";
    }

    auto ulFileSize = static_cast<size_t>(fileSize);

    try {
        TBlob<uint8_t>::Ptr weightsPtr(new TBlob<uint8_t>(TensorDesc(Precision::U8, {ulFileSize}, Layout::C)));
        weightsPtr->allocate();
        FileUtils::readAllFile(filepath, weightsPtr->buffer(), ulFileSize);
        return SetWeights(weightsPtr, resp);
    } catch (const InferenceEngineException& ex) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      } catch (const InferenceEngineException& ex) {" << std::endl;
        return DescriptionBuffer(resp) << ex.what();
    }
}

StatusCode CNNNetReaderImpl::ReadNetwork(const char* filepath, ResponseDesc* resp) noexcept {
    IE_PROFILING_AUTO_SCOPE(CNNNetReaderImpl::ReadNetwork)
    if (network) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      if (network) {" << std::endl;
        return DescriptionBuffer(NETWORK_NOT_READ, resp)
               << "Network has been read already, use new reader instance to read new network.";
    }

    auto parse_result = ParseXml(filepath);
    if (!parse_result.error_msg.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      if (!parse_result.error_msg.empty()) {" << std::endl;
        return DescriptionBuffer(resp) << parse_result.error_msg;
    }
    xmlDoc = std::move(parse_result.xml);

    StatusCode ret = ReadNetwork();
    if (ret != OK) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      if (ret != OK) {" << std::endl;
        return DescriptionBuffer(resp) << "Error reading network: " << description;
    }
    return OK;
}

StatusCode CNNNetReaderImpl::ReadNetwork() {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:  StatusCode CNNNetReaderImpl::ReadNetwork() {" << std::endl;
    description.clear();

    try {
        // check which version it is...
        pugi::xml_node root = xmlDoc->document_element();

        _version = GetFileVersion(root);
        if (_version < 2) THROW_IE_EXCEPTION << "deprecated IR version: " << _version;
        if (_version == 10) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:          if (_version == 10) {" << std::endl;
            // Activate an alternative code path for V10 that should be read into ngraph::Function
            // We cannot proceed with reading right now, because there is not binary file loaded.
            // So we are postponing real read until weights are specified.
            parseSuccess = true;
        } else if (_version < 10) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:          } else if (_version < 10) {" << std::endl;
            _parser = parserCreator->create(_version);
            InferenceEngine::details::CNNNetworkImplPtr local_network = _parser->Parse(root);
            name = local_network->getName();
            local_network->validate(_version);
            network = local_network;
            parseSuccess = true;
        } else {
            THROW_IE_EXCEPTION << "cannot parse future versions: " << _version;
        }
    } catch (const std::string& err) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      } catch (const std::string& err) {" << std::endl;
        description = err;
        parseSuccess = false;
        return GENERAL_ERROR;
    } catch (const InferenceEngineException& e) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      } catch (const InferenceEngineException& e) {" << std::endl;
        description = e.what();
        parseSuccess = false;
        return GENERAL_ERROR;
    } catch (const std::exception& e) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      } catch (const std::exception& e) {" << std::endl;
        description = e.what();
        parseSuccess = false;
        return GENERAL_ERROR;
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:      } catch (...) {" << std::endl;
        description = "Unknown exception thrown";
        parseSuccess = false;
        return UNEXPECTED;
    }

    return OK;
}

void CNNNetReaderImpl::addExtensions(const std::vector<InferenceEngine::IExtensionPtr>& ext) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:  void CNNNetReaderImpl::addExtensions(const std::vector<InferenceEngine::IExtensionPtr>& ext) {" << std::endl;
    extensions = ext;
}

std::shared_ptr<IFormatParser> V2FormatParserCreator::create(size_t version) {
    std::cerr << "./inference-engine/src/inference_engine/ie_cnn_net_reader_impl.cpp:  std::shared_ptr<IFormatParser> V2FormatParserCreator::create(size_t version) {" << std::endl;
#ifdef ENABLE_IR_READER
    return std::make_shared<FormatParser>(version);
#else
    THROW_IE_EXCEPTION << "Please, recompile Inference Engine library with the ENABLE_IR_READER=ON Cmake option";
    return nullptr;
#endif
}

InferenceEngine::ICNNNetReader* InferenceEngine::CreateCNNNetReader() noexcept {
    return new CNNNetReaderImpl(std::make_shared<V2FormatParserCreator>());
}
IE_SUPPRESS_DEPRECATED_END
