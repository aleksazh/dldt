// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <sched.h>
#include "lin_system_conf.h"
#include "ie_parallel.hpp"


namespace MKLDNNPlugin {
namespace cpu {

Processor::Processor() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  Processor::Processor() {" << std::endl;
    processor = 0;
    physicalId = 0;
    cpuCores = 0;
}

CpuInfo::CpuInfo() : fileContentBegin(nullptr) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  CpuInfo::CpuInfo() : fileContentBegin(nullptr) {" << std::endl;
    //loadContentFromFile("/proc/cpuinfo");
    loadContentFromFile("./cpuinfo");
}

void CpuInfo::loadContentFromFile(const char *fileName) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  void CpuInfo::loadContentFromFile(const char *fileName) {" << std::endl;
    std::ifstream file(fileName);
    std::string content(
            (std::istreambuf_iterator<char>(file)),
            (std::istreambuf_iterator<char>()));

    loadContent(content.c_str());
}

void CpuInfo::loadContent(const char *content) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  void CpuInfo::loadContent(const char *content) {" << std::endl;
    size_t contentLength = strlen(content);
    char *contentCopy = new char[contentLength + 1];
    snprintf(contentCopy, contentLength + 1, "%s", content);

    parseLines(contentCopy);

    if (fileContentBegin != nullptr)
        delete[] fileContentBegin;
    fileContentBegin = contentCopy;
    fileContentEnd = &contentCopy[contentLength];
    currentLine = NULL;
}

CpuInfo::~CpuInfo() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  CpuInfo::~CpuInfo() {" << std::endl;
    if (fileContentBegin != nullptr)
        delete[] fileContentBegin;
}

void CpuInfo::parseLines(char *content) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  void CpuInfo::parseLines(char *content) {" << std::endl;
    for (; *content; content++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      for (; *content; content++) {" << std::endl;
        if (*content == '\n') {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:          if (*content == '\n') {" << std::endl;
            *content = '\0';
        }
    }
}

const char *CpuInfo::getFirstLine() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  const char *CpuInfo::getFirstLine() {" << std::endl;
    currentLine = fileContentBegin < fileContentEnd ? fileContentBegin : NULL;
    return getNextLine();
}

const char *CpuInfo::getNextLine() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  const char *CpuInfo::getNextLine() {" << std::endl;
    if (!currentLine) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      if (!currentLine) {" << std::endl;
        return NULL;
    }

    const char *savedCurrentLine = currentLine;
    while (*(currentLine++)) {
    }

    if (currentLine >= fileContentEnd) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      if (currentLine >= fileContentEnd) {" << std::endl;
        currentLine = NULL;
    }

    return savedCurrentLine;
}

Collection::Collection(CpuInfoInterface *cpuInfo) : cpuInfo(*cpuInfo) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  Collection::Collection(CpuInfoInterface *cpuInfo) : cpuInfo(*cpuInfo) {" << std::endl;
    totalNumberOfSockets = 0;
    totalNumberOfCpuCores = 0;
    currentProcessor = NULL;

    processors.reserve(96);

    parseCpuInfo();
    collectBasicCpuInformation();
}

unsigned Collection::getTotalNumberOfSockets() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  unsigned Collection::getTotalNumberOfSockets() {" << std::endl;
    return totalNumberOfSockets;
}

unsigned Collection::getTotalNumberOfCpuCores() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  unsigned Collection::getTotalNumberOfCpuCores() {" << std::endl;
    return totalNumberOfCpuCores;
}

unsigned Collection::getNumberOfProcessors() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  unsigned Collection::getNumberOfProcessors() {" << std::endl;
    return processors.size();
}

void Collection::parseCpuInfo() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  void Collection::parseCpuInfo() {" << std::endl;
    const char *cpuInfoLine = cpuInfo.getFirstLine();
    for (; cpuInfoLine; cpuInfoLine = cpuInfo.getNextLine()) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      for (; cpuInfoLine; cpuInfoLine = cpuInfo.getNextLine()) {" << std::endl;
        parseCpuInfoLine(cpuInfoLine);
    }
}

void Collection::parseCpuInfoLine(const char *cpuInfoLine) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  void Collection::parseCpuInfoLine(const char *cpuInfoLine) {" << std::endl;
    int delimiterPosition = strcspn(cpuInfoLine, ":");

    if (cpuInfoLine[delimiterPosition] == '\0') {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      if (cpuInfoLine[delimiterPosition] == '\0') {" << std::endl;
        currentProcessor = NULL;
    } else {
        parseValue(cpuInfoLine, &cpuInfoLine[delimiterPosition + 2]);
    }
}

void Collection::parseValue(const char *fieldName, const char *valueString) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  void Collection::parseValue(const char *fieldName, const char *valueString) {" << std::endl;
    if (!currentProcessor) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      if (!currentProcessor) {" << std::endl;
        appendNewProcessor();
    }

    if (beginsWith(fieldName, "processor")) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      if (beginsWith(fieldName, 'processor')) {" << std::endl;
        currentProcessor->processor = parseInteger(valueString);
    }

    if (beginsWith(fieldName, "physical id")) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      if (beginsWith(fieldName, 'physical id')) {" << std::endl;
        currentProcessor->physicalId = parseInteger(valueString);
    }

    if (beginsWith(fieldName, "cpu cores")) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      if (beginsWith(fieldName, 'cpu cores')) {" << std::endl;
        currentProcessor->cpuCores = parseInteger(valueString);
    }
}

void Collection::appendNewProcessor() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  void Collection::appendNewProcessor() {" << std::endl;
    processors.push_back(Processor());
    currentProcessor = &processors.back();
}

bool Collection::beginsWith(const char *lineBuffer, const char *text) const {
    while (*text) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      while (*text) {" << std::endl;
        if (*(lineBuffer++) != *(text++)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:          if (*(lineBuffer++) != *(text++)) {" << std::endl;
            return false;
        }
    }

    return true;
}

unsigned Collection::parseInteger(const char *text) const {
    return atol(text);
}

void Collection::collectBasicCpuInformation() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  void Collection::collectBasicCpuInformation() {" << std::endl;
    std::set<unsigned> uniquePhysicalId;
    std::vector<Processor>::iterator processor = processors.begin();
    for (; processor != processors.end(); processor++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      for (; processor != processors.end(); processor++) {" << std::endl;
        uniquePhysicalId.insert(processor->physicalId);
        updateCpuInformation(*processor, uniquePhysicalId.size());
    }
}

void Collection::updateCpuInformation(const Processor &processor,
                                      unsigned numberOfUniquePhysicalId) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp: unsigned numberOfUniquePhysicalId) {" << std::endl;
    if (totalNumberOfSockets == numberOfUniquePhysicalId) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      if (totalNumberOfSockets == numberOfUniquePhysicalId) {" << std::endl;
        return;
    }

    totalNumberOfSockets = numberOfUniquePhysicalId;
    totalNumberOfCpuCores += processor.cpuCores;
}

#if !((IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO))
    std::vector<int> getAvailableNUMANodes() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      std::vector<int> getAvailableNUMANodes() {" << std::endl;
    static CpuInfo cpuInfo;
    static Collection collection(&cpuInfo);
    std::vector<int> nodes;
    for (int i = 0; i < collection.getTotalNumberOfSockets(); i++)
        nodes.push_back(i);
    return nodes;
}
#endif

int getNumberOfCPUCores() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:  int getNumberOfCPUCores() {" << std::endl;
    static CpuInfo cpuInfo;
    static Collection collection(&cpuInfo);
    unsigned numberOfProcessors = collection.getNumberOfProcessors();
    unsigned totalNumberOfCpuCores = collection.getTotalNumberOfCpuCores();
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp: numberOfProcessors: " << numberOfProcessors << ", totalNumberOfCpuCores: " << totalNumberOfCpuCores << std::endl;

    cpu_set_t usedCoreSet, currentCoreSet, currentCpuSet;
    CPU_ZERO(&currentCpuSet);
    CPU_ZERO(&usedCoreSet);
    CPU_ZERO(&currentCoreSet);

    sched_getaffinity(0, sizeof(currentCpuSet), &currentCpuSet);

    for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:      for (int processorId = 0; processorId < numberOfProcessors; processorId++) {" << std::endl;
        if (CPU_ISSET(processorId, &currentCpuSet)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:          if (CPU_ISSET(processorId, &currentCpuSet)) {" << std::endl;
            unsigned coreId = processorId % totalNumberOfCpuCores;
            if (!CPU_ISSET(coreId, &usedCoreSet)) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn/os/lin/lin_system_conf.cpp:              if (!CPU_ISSET(coreId, &usedCoreSet)) {" << std::endl;
                CPU_SET(coreId, &usedCoreSet);
                CPU_SET(processorId, &currentCoreSet);
            }
        }
    }
    return CPU_COUNT(&currentCoreSet);
}

}  // namespace cpu
}  // namespace MKLDNNPlugin
