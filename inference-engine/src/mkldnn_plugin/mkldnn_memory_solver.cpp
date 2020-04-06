#include <iostream>
// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_memory_solver.hpp"

#include <details/ie_exception.hpp>

#include <algorithm>
#include <vector>
#include <map>

namespace MKLDNNPlugin {

MemorySolver::MemorySolver(const std::vector<Box>& boxes) : _boxes(boxes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:  MemorySolver::MemorySolver(const std::vector<Box>& boxes) : _boxes(boxes) {" << std::endl;
    int max_ts = 0;
    for (const Box &box : _boxes) max_ts = std::max(std::max(max_ts, box.start), box.finish);
    for (Box &box : _boxes) if (box.finish == -1) box.finish = max_ts;

    // sort by start and finish ts
    std::sort(_boxes.begin(), _boxes.end(), [](const Box& l, const Box& r) -> bool
        { return l.start < r.start || (l.start == r.start && l.finish < r.finish); });

    // remove unused timestamps (not a begin of some box)
    // each ts should start a box
    std::vector<bool> ts_exist(max_ts+1);
    for (const Box &b : _boxes) ts_exist[b.start] = true;

    int rm_ts_s = 0, rm_ts_f = 0;
    int ts_s = 0, ts_f = 0;
    for (Box &b : _boxes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:      for (Box &b : _boxes) {" << std::endl;
        while (ts_s < b.start) if (!ts_exist[ts_s++]) rm_ts_s++;

        if (ts_f > b.finish + 1) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:          if (ts_f > b.finish + 1) {" << std::endl; ts_f = ts_s; rm_ts_f = rm_ts_s; }
        while (ts_f <= b.finish) if (!ts_exist[ts_f++]) rm_ts_f++;

        b.start -= rm_ts_s;
        b.finish -= rm_ts_f;
    }
    _time_duration = ts_f - rm_ts_f;
}

inline bool popupTogetherWith(MemorySolver::Box &box_new, const MemorySolver::Box &box_old) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:  inline bool popupTogetherWith(MemorySolver::Box &box_new, const MemorySolver::Box &box_old) {" << std::endl;
    if (box_new.id+box_new.size > box_old.id &&
        box_old.id+box_old.size > box_new.id) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:          box_old.id+box_old.size > box_new.id) {" << std::endl;
        // Move the new one up. There is an intersection
        box_new.id = box_old.id + box_old.size;
        return true;
    } else {
        return false;
    }
}

int64_t MemorySolver::solve() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:  int64_t MemorySolver::solve() {" << std::endl;
    maxTopDepth();  // at first make sure that we no need more for boxes sorted by box.start
    std::vector<std::vector<const Box*>> time_slots(_time_duration);
    for (auto & slot : time_slots) slot.reserve(_top_depth);  // 2D array [_time_duration][_top_depth]

    // Sort be box size. First is biggest
    // Comment this line to check other order of box putting
    std::sort(_boxes.begin(), _boxes.end(), [](const Box& l, const Box& r)
        {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:      std::sort(_boxes.begin(), _boxes.end(), [](const Box& l, const Box& r)         {" << std::endl; return l.size > r.size; });

    int64_t _min_required = 0;

    for (Box& box : _boxes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:      for (Box& box : _boxes) {" << std::endl;
        // start from bottom and will lift it up if intersect with other present
        int64_t id = box.id;
        box.id = 0;  // id will be used as a temp offset storage
        bool popped_up;
        do {
            popped_up = false;
            for (int i_slot = box.start; i_slot <= box.finish; i_slot++) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:              for (int i_slot = box.start; i_slot <= box.finish; i_slot++) {" << std::endl;
                for (auto *box_in_slot : time_slots[i_slot]) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:                  for (auto *box_in_slot : time_slots[i_slot]) {" << std::endl;
                    // intersect with already stored boxes for all covered time slots
                    // and move up the new one if needed
                    popped_up |= popupTogetherWith(box, *box_in_slot);
                }
            }
        } while (popped_up);

        // add current box to covered time slot
        for (int i_slot = box.start; i_slot <= box.finish; i_slot++)
            time_slots[i_slot].push_back(&box);

        // store the max top bound for each box
        _min_required = std::max(_min_required, box.id + box.size);
        _offsets[id] = box.id;
    }

    return _min_required;
}

int64_t MemorySolver::maxDepth() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:  int64_t MemorySolver::maxDepth() {" << std::endl;
    if (_depth == -1) calcDepth();
    return _depth;
}

int64_t MemorySolver::maxTopDepth() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:  int64_t MemorySolver::maxTopDepth() {" << std::endl;
    if (_top_depth == -1) calcDepth();
    return _top_depth;
}

int64_t MemorySolver::getOffset(int id) const {
    auto res = _offsets.find(id);
    if (res == _offsets.end()) THROW_IE_EXCEPTION << "There are no box for provided ID";
    return res->second;
}

//======== Private =============//

void MemorySolver::calcDepth() {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:  void MemorySolver::calcDepth() {" << std::endl;
    int64_t top_depth = 0;
    int64_t depth = 0;
    std::map<int64_t, std::vector<const Box*>> release_at;

    for (const Box& box : _boxes) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:      for (const Box& box : _boxes) {" << std::endl;
        int64_t time = box.start;
        depth += box.size;
        top_depth++;

        release_at[box.finish+1].push_back(&box);

        for (const Box *b : release_at[time]) {
    std::cerr << "./inference-engine/src/mkldnn_plugin/mkldnn_memory_solver.cpp:          for (const Box *b : release_at[time]) {" << std::endl;
            depth -= b->size;
            top_depth--;
        }
        release_at.erase(time);
        IE_ASSERT(top_depth > 0);

        _top_depth = std::max(_top_depth, top_depth);
        _depth = std::max(_depth, depth);
    }
}

}  // namespace MKLDNNPlugin
