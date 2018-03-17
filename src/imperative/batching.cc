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
#include <mxnet/batching.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace mxnet {
#if DMLC_CXX11_THREAD_LOCAL
thread_local bool DBatchEngine::is_dbatch_ = false;
#else
MX_THREAD_LOCAL bool DBatchEngine::is_dbatch_ = false;
#endif

DBatchEngine* DBatchEngine::Get() {
  static DBatchEngine inst;
  return &inst;
}

int64_t ComputeNodeSign(const nnvm::NodePtr& n) {
  static int cnt = 1;
  // (TODO szha)
  if (n->is_variable()) {
    return cnt++;
  }
  return reinterpret_cast<int64_t>(n->op());
}

nnvm::Graph DBatchEngine::BatchGraphs(const std::vector<nnvm::Graph>& graphs) {
  // collect depth
  int max_depth = 0;
  std::unordered_map<nnvm::Node*, int> depth_map;
  std::unordered_set<nnvm::NodeEntry, nnvm::NodeEntryHash, nnvm::NodeEntryEqual> prev_outputs;
  for (const nnvm::Graph& g : graphs) {
    std::vector<nnvm::NodeEntry> g_outputs = g.outputs;
    LOG(INFO) << "graph outputs " << g_outputs.size();
    std::copy(g_outputs.begin(), g_outputs.end(), std::inserter(prev_outputs, prev_outputs.end()));
    nnvm::DFSVisit(g_outputs, [&](const nnvm::NodePtr& n){
      int depth = 0;
      for (auto e : n->inputs) {
        int idepth = depth_map.at(e.node.get());
        depth = std::max(depth, idepth+1);
      }
      LOG(INFO) << "depth of (" << n->attrs.name << "): " << depth;
      depth_map.insert({n.get(), depth});
      max_depth = std::max(depth, max_depth);
    });
  }
  LOG(INFO) << "total graph outputs " << prev_outputs.size();

  LOG(INFO) << "max depth " << max_depth;
  // depth: sign->node
  std::vector<std::unordered_map<int64_t, std::vector<nnvm::NodePtr>>> forward_steps;
  forward_steps.resize(max_depth + 1);

  for (const nnvm::Graph& g : graphs) {
    nnvm::DFSVisit(g.outputs, [&](const nnvm::NodePtr& n){
      int64_t sign = ComputeNodeSign(n);
      int depth = depth_map.at(n.get());
      forward_steps[depth][sign].emplace_back(n);
    });
  }

  // generate new graph
  // create empty mapping from old node to new node
  // for each step
    // for each op
      // update inputs according to node mapping
      // concat inputs and bookkeep lengths
      // create op with concat inputs
      // slice according to lengths to get new node
      // record mapping from old node to new node
  // (TODO szha)

  return nnvm::Graph();
}

void DBatchEngine::ExecuteGraph(const nnvm::Graph& graph) {
  // TODO
}

}  // namespace mxnet
