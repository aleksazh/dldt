#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <xml_parse_utils.h>

#include <ie_ir_reader.hpp>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <string>
#include <vector>

#include "description_buffer.hpp"
#include "ie_ir_parser.hpp"
#include "ie_ngraph_utils.hpp"

using namespace InferenceEngine;

static size_t GetIRVersion(pugi::xml_node& root) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:  static size_t GetIRVersion(pugi::xml_node& root) {" << std::endl;
    return XMLParseUtils::GetUIntAttr(root, "version", 0);
}

std::shared_ptr<ngraph::Function> IRReader::read(const std::string& modelPath, const std::string& binPath) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:  std::shared_ptr<ngraph::Function> IRReader::read(const std::string& modelPath, const std::string& binPath) {" << std::endl;
    std::ifstream modelFile(modelPath);
    if (!modelFile.is_open()) THROW_IE_EXCEPTION << "File " << modelPath << " cannot be openned!";

    std::stringstream modelBuf;
    modelBuf << modelFile.rdbuf();

    Blob::Ptr weights;
    std::string bPath = binPath;
    if (bPath.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:      if (bPath.empty()) {" << std::endl;
        bPath = modelPath;
        auto pos = bPath.rfind('.');
        if (pos != std::string::npos) bPath = bPath.substr(0, pos);
        bPath += ".bin";

        if (!FileUtils::fileExist(bPath)) bPath.clear();
    }

    if (!bPath.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:      if (!bPath.empty()) {" << std::endl;
        int64_t fileSize = FileUtils::fileSize(bPath);

        if (fileSize < 0)
            THROW_IE_EXCEPTION << "Filesize for: " << bPath << " - " << fileSize
                               << " < 0. Please, check weights file existence.";

        size_t ulFileSize = static_cast<size_t>(fileSize);

        weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {ulFileSize}, Layout::C));
        weights->allocate();
        FileUtils::readAllFile(bPath, weights->buffer(), ulFileSize);
    }

    return read(modelBuf.str(), weights);
}

std::shared_ptr<ngraph::Function> IRReader::read(const std::string& model, const Blob::CPtr& weights) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:  std::shared_ptr<ngraph::Function> IRReader::read(const std::string& model, const Blob::CPtr& weights) {" << std::endl;
    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_buffer(model.data(), model.length());
    if (res.status != pugi::status_ok) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:      if (res.status != pugi::status_ok) {" << std::endl;
        THROW_IE_EXCEPTION << res.description() << "at offset " << res.offset;
    }
    return readXml(xmlDoc, weights);
}

std::shared_ptr<ngraph::Function> IRReader::readXml(const pugi::xml_document& xmlDoc, const Blob::CPtr& weights) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:  std::shared_ptr<ngraph::Function> IRReader::readXml(const pugi::xml_document& xmlDoc, const Blob::CPtr& weights) {" << std::endl;
    try {
        // check which version it is...
        pugi::xml_node root = xmlDoc.document_element();

        auto version = GetIRVersion(root);
        IRParser parser(version, extensions);
        return parser.parse(root, weights);
    } catch (const std::string& err) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:      } catch (const std::string& err) {" << std::endl;
        THROW_IE_EXCEPTION << err;
    } catch (const details::InferenceEngineException& e) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:      } catch (const details::InferenceEngineException& e) {" << std::endl;
        throw;
    } catch (const std::exception& e) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:      } catch (const std::exception& e) {" << std::endl;
        THROW_IE_EXCEPTION << e.what();
    } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/ie_ir_reader.cpp:      } catch (...) {" << std::endl;
        THROW_IE_EXCEPTION << "Unknown exception thrown";
    }
}
