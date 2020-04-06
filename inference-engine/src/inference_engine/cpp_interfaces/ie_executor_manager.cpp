#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/ie_executor_manager.hpp"
#include "cpp_interfaces/ie_task_executor.hpp"

#include <memory>
#include <string>

namespace InferenceEngine {

ITaskExecutor::Ptr ExecutorManagerImpl::getExecutor(std::string id) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp:  ITaskExecutor::Ptr ExecutorManagerImpl::getExecutor(std::string id) {" << std::endl;
    auto foundEntry = executors.find(id);
    if (foundEntry == executors.end()) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp:      if (foundEntry == executors.end()) {" << std::endl;
        auto newExec = std::make_shared<TaskExecutor>(id);
        executors[id] = newExec;
        return newExec;
    }
    return foundEntry->second;
}

// for tests purposes
size_t ExecutorManagerImpl::getExecutorsNumber() {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp:  size_t ExecutorManagerImpl::getExecutorsNumber() {" << std::endl;
    return executors.size();
}

void ExecutorManagerImpl::clear() {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp:  void ExecutorManagerImpl::clear() {" << std::endl;
    executors.clear();
}

ExecutorManager* ExecutorManager::_instance = nullptr;

ITaskExecutor::Ptr ExecutorManager::getExecutor(std::string id) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp:  ITaskExecutor::Ptr ExecutorManager::getExecutor(std::string id) {" << std::endl;
    return _impl.getExecutor(id);
}

size_t ExecutorManager::getExecutorsNumber() {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp:  size_t ExecutorManager::getExecutorsNumber() {" << std::endl;
    return _impl.getExecutorsNumber();
}

void ExecutorManager::clear() {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_executor_manager.cpp:  void ExecutorManager::clear() {" << std::endl;
    _impl.clear();
}

}  // namespace InferenceEngine
