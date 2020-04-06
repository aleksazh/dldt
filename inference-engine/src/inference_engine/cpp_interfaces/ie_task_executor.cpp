#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/ie_task_executor.hpp"

#include <condition_variable>
#include <ie_profiling.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "details/ie_exception.hpp"

namespace InferenceEngine {

TaskExecutor::TaskExecutor(std::string name): _isStopped(false), _name(name) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:  TaskExecutor _name: " << _name.c_str() << std::endl;
    /*_thread = std::make_shared<std::thread>([&] {
        std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp: annotateSetThreadName(('TaskExecutor thread for ' + _name).c_str());" << std::endl;
        annotateSetThreadName(("TaskExecutor thread for " + _name).c_str());
        std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp: while (!_isStopped) {" << std::endl;
        while (!_isStopped) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:          while (!_isStopped) {" << std::endl;
            bool isQueueEmpty;
            Task currentTask;
            {  // waiting for the new task or for stop signal
                std::unique_lock<std::mutex> lock(_queueMutex);
                _queueCondVar.wait(lock, [&]() {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:                  _queueCondVar.wait(lock, [&]() {" << std::endl;
                    return !_taskQueue.empty() || _isStopped;
                });
                isQueueEmpty = _taskQueue.empty();
                if (!isQueueEmpty) currentTask = _taskQueue.front();
            }
            if (_isStopped && isQueueEmpty) break;
            if (!isQueueEmpty) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:              if (!isQueueEmpty) {" << std::endl;
                currentTask();
                std::unique_lock<std::mutex> lock(_queueMutex);
                _taskQueue.pop();
                isQueueEmpty = _taskQueue.empty();
                if (isQueueEmpty) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:                  if (isQueueEmpty) {" << std::endl;
                    // notify dtor, that all tasks were completed
                    _queueCondVar.notify_all();
                }
            }
        }
    });*/
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp: TaskExecutor end" << std::endl;
}

TaskExecutor::~TaskExecutor() {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:  TaskExecutor::~TaskExecutor() {" << std::endl;
    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        if (!_taskQueue.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:          if (!_taskQueue.empty()) {" << std::endl;
            _queueCondVar.wait(lock, [this]() {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:              _queueCondVar.wait(lock, [this]() {" << std::endl;
                return _taskQueue.empty();
            });
        }
        _isStopped = true;
        _queueCondVar.notify_all();
    }
    if (_thread && _thread->joinable()) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:      if (_thread && _thread->joinable()) {" << std::endl;
        _thread->join();
        _thread.reset();
    }
}

void TaskExecutor::run(Task task) {
    std::cerr << "./inference-engine/src/inference_engine/cpp_interfaces/ie_task_executor.cpp:  void TaskExecutor::run(Task task) {" << std::endl;
    std::unique_lock<std::mutex> lock(_queueMutex);
    _taskQueue.push(std::move(task));
    _queueCondVar.notify_all();
}

}  // namespace InferenceEngine
