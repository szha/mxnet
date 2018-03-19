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

nnvm::NodeEntry MapNodeEntry(const nnvm::NodeEntry& entry,
                             const std::unordered_map<nnvm::Node*,
                                                      std::vector<nnvm::NodePtr>>& node_map) {
  nnvm::Node* node_ptr = entry.node.get();
  auto found_node = node_map.find(node_ptr);
  if (found_node == node_map.end()) {
    return entry;
  }
  std::vector<nnvm::NodePtr> new_nodes = found_node->second;
  nnvm::NodePtr result_node;
  // when unbatchable node was processed, only one entry is added. because it's not NodeEntry, it's
  // not possible to replicate the nodes upfront.
  result_node = new_nodes[entry.index < new_nodes.size() ? entry.index : 0]; // possibly non-batchable multi-out node
  return nnvm::NodeEntry{result_node, 0, entry.version+1};
}

nnvm::NodePtr& MapNode(const nnvm::NodePtr& node,
                       const std::unordered_map<nnvm::Node*,
                                                std::vector<nnvm::NodePtr>>& node_map) {
  nnvm::Node* node_ptr = node.get();
  std::vector<nnvm::NodeEntry>& old_inputs = node_ptr->inputs;
  std::vector<nnvm::NodeEntry> new_inputs(old_inputs.size());
  for (const nnvm::NodeEntry input_entry : old_inputs) {
    new_inputs.emplace_back(MapNodeEntry(input_entry, node_map));
  }
  nnvm::NodePtr new_node = nnvm::Node::Create();
  new_node->attrs = node_ptr->attrs;
  new_node->inputs = new_inputs;
  CHECK_EQ(node_ptr->control_deps.size(), 0); // TODO map control_deps node ptr
  new_node->info = node_ptr->info;

  return new_node;
}

nnvm::NodeEntry CreateConcatNode(const std::vector<nnvm::NodePtr>& op_ptrs,
                                 uint32_t input_index,
                                 const std::unordered_map<nnvm::Node*, std::vector<nnvm::NodePtr>>& node_map) {
  size_t num_nodes = op_ptrs.size();
  size_t in_batch_axis = 0; // TODO get batch axis from nodes for each input index

  // batchable input, create concat node
  std::vector<nnvm::NodeEntry> node_inputs(num_nodes);
  std::string node_name = "batch_input_" + std::to_string(input_index) + "_concat";
  for (const nnvm::NodePtr op_ptr : op_ptrs) {
    nnvm::NodeEntry input_entry = op_ptr.get()->inputs[input_index];
    node_inputs.emplace_back(MapNodeEntry(input_entry, node_map));
    node_name += "_" + input_entry.node->attrs.name;
  }
  std::unordered_map<std::string, std::string> concat_args = {
    {"num_args", std::to_string(num_nodes)},
    {"dim", std::to_string(in_batch_axis)}
  };

  return MakeNode("concat", node_name, node_inputs, concat_args);
}

nnvm::Graph DBatchEngine::BatchGraphs(const std::vector<nnvm::Graph>& graphs) {
  // collect depth
  int max_depth = 0;
  std::unordered_map<nnvm::Node*, int> depth_map;
  std::vector<nnvm::NodeEntry> prev_outputs;
  for (const nnvm::Graph& g : graphs) {
    const std::vector<nnvm::NodeEntry>& g_outputs = g.outputs;
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
  // create mapping from old node to new node
  std::unordered_map<nnvm::Node*, std::vector<nnvm::NodePtr>> old_new_node_map;
  for (uint32_t istep = 0; istep < forward_steps.size(); istep++) {

    const std::unordered_map<int64_t, std::vector<nnvm::NodePtr>> step = forward_steps[istep];
    for (const std::pair<int64_t, std::vector<nnvm::NodePtr>> step_ops : step) {

      int64_t op_sign = step_ops.first;
      const std::vector<nnvm::NodePtr>& op_ptrs = step_ops.second;
      size_t num_nodes = op_ptrs.size();
      nnvm::Node* first_op_node = op_ptrs.front().get();

      if (num_nodes == 1) { // op that can't be batched, record mapping and move on
        old_new_node_map[first_op_node].emplace_back(MapNode(op_ptrs.front(), old_new_node_map));
        continue;
      }

      size_t num_inputs = first_op_node->num_inputs(), num_outputs = first_op_node->num_outputs();

      std::unordered_set<size_t> batchable_data_indices = {0}; // TODO find batchable data tensors from op

      // record lengths of each sample from the first batchable data's input batch axis
      std::vector<size_t> sample_lengths(num_nodes);
      size_t in_batch_axis = 0; // TODO get batch axis from nodes for each input index
      for (uint32_t j = 0; j < num_nodes; j++) {
        const nnvm::NodePtr op_ptr = op_ptrs[j];
        size_t first_batchable_data_index = *batchable_data_indices.begin();
        auto& first_batchable_input = op_ptr.get()->inputs[first_batchable_data_index];
        NDArray& entry_ndarray = entry_arr_[first_batchable_input];
        const TShape& shape = entry_ndarray.shape();
        sample_lengths[j] = shape[in_batch_axis];
      }

      std::vector<nnvm::NodeEntry> batch_node_inputs(num_inputs);
      for (uint32_t i = 0; i < num_inputs; i++) {
        if (batchable_data_indices.find(i) != batchable_data_indices.end()) {

          batch_node_inputs.emplace_back(CreateConcatNode(op_ptrs, i, old_new_node_map));
        } else { // TODO unbatchable input, signature should not allow different unbatchable inputs
          batch_node_inputs.emplace_back(MapNodeEntry(first_op_node->inputs[i],
                                                      old_new_node_map));
        }
      }

      // create op with concat inputs
      const Op* batch_op = first_op_node->op();
      nnvm::NodeEntry batch_node = MakeNode(batch_op->name.c_str(),
                                            "batch_"+std::to_string(op_sign),
                                            batch_node_inputs, first_op_node->attrs.dict);

      for (uint32_t i = 0; i < num_outputs; i++) {
        nnvm::NodeEntry out_node{batch_node.node, i, batch_node.version+i+1};
        std::vector<nnvm::NodeEntry> slice_input = {out_node};

        // slice according to lengths to get new node
        for (uint32_t j = 0, begin = 0; j < num_nodes; j++, begin += sample_lengths[j]) {
          size_t out_batch_axis = 0; // TODO get batch axis for each output
          std::unordered_map<std::string, std::string> slice_args = {
            {"begin", "(" + std::to_string(begin) + ",)"},
            {"end", "(" + std::to_string(begin+sample_lengths[j]) + ",)"}
          };
          nnvm::NodeEntry slice_node = MakeNode("slice", batch_node.node.get()->attrs.name,
                                                slice_input, slice_args);
          old_new_node_map[op_ptrs[j].get()].emplace_back(slice_node.node);
        }
      }
    }
  }
  std::vector<nnvm::NodeEntry> new_outputs(prev_outputs.size());
  for (const nnvm::NodeEntry prev_out : prev_outputs) {
    new_outputs.emplace_back(MapNodeEntry(prev_out, old_new_node_map));
  }
  nnvm::Graph new_graph;
  new_graph.outputs = new_outputs;
  // Print graph
  {
    nnvm::Symbol sym;
    sym.outputs = prev_outputs;
    std::cout << "Full Old Graph: \n";
    sym.Print(std::cout);
    sym.outputs = new_graph.outputs;
    std::cout << "Full New Graph: \n";
    sym.Print(std::cout);
  }

  return new_graph;
}

void DBatchEngine::ExecuteGraph(const nnvm::Graph& graph) {
  // TODO
}

}  // namespace mxnet
