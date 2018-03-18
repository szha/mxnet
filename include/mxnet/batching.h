/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef MXNET_BATCHING_H_
#define MXNET_BATCHING_H_

#include <mxnet/op_attr_types.h>
#include <mxnet/graph_attr_types.h>
#include <nnvm/symbolic.h>
#include <nnvm/op.h>
#include <nnvm/graph.h>
#include <vector>
#include <utility>
#include <string>
#include <unordered_map>

#include "./ndarray.h"

namespace mxnet {

class DBatchEngine {
 public:
  void SaveGraph(nnvm::Graph& g) {
    graphs_.push_back(g);
  }

  const std::vector<nnvm::Graph>& Graphs() const {
    return graphs_;
  }

  void Batch() {
    // (TODO szha) batch and execution
    LOG(INFO) << "batching, graphs size=" << graphs_.size();
    nnvm::Graph g = BatchGraphs(graphs_);
    ExecuteGraph(g);
  }

  void Fresh() {
    graphs_.clear();
  }

  void RecordArray(const NDArray& arr) {
    if (entry_arr_.count(arr.entry_)) {
      CHECK_EQ(entry_arr_[arr.entry_].var(), arr.var());
    } else {
      entry_arr_[arr.entry_] = arr;
    }
  }

  bool is_dbatch() const {
    return is_dbatch_;
  }

  bool set_is_dbatch(bool is_dbatch) {
    is_dbatch_ = is_dbatch;
    return is_dbatch_;
  }

  static DBatchEngine* Get();

 private:
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local bool is_dbatch_;
#else
  static MX_THREAD_LOCAL bool is_dbatch_;
#endif
  std::vector<nnvm::Graph> graphs_;
  nnvm::NodeEntryMap<NDArray> entry_arr_;

  nnvm::Graph BatchGraphs(std::vector<nnvm::Graph>& graphs);
  void ExecuteGraph(nnvm::Graph& graph);
};

}  // namespace mxnet

#endif
