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
#include <mxnet/imperative.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "./imperative_utils.h"

namespace mxnet {
#if DMLC_CXX11_THREAD_LOCAL
thread_local bool DBatchEngine::is_dbatch_ = false;
thread_local int DBatchEngine::batch_size_ = 0;
#else
MX_THREAD_LOCAL bool DBatchEngine::is_dbatch_ = false;
MX_THREAD_LOCAL bool DBatchEngine::batch_size_ = 0;
#endif

DBatchEngine* DBatchEngine::Get() {
  static DBatchEngine inst;
  return &inst;
}

uint64_t ComputeNodeSign(const nnvm::NodePtr& n) {
  static int cnt = 1;
  // (TODO szha)
  if (n->is_variable()) {
    return cnt++;
  }
  return reinterpret_cast<uint64_t>(n->op());
}

nnvm::NodeEntry MapNodeEntry(const nnvm::NodeEntry& entry,
                             const std::unordered_map<nnvm::Node*,
                                                      std::vector<nnvm::NodePtr>>& node_map) {
  nnvm::Node* node_ptr = entry.node.get();
  auto found_node = node_map.find(node_ptr);
  CHECK(found_node != node_map.end());
  std::vector<nnvm::NodePtr> new_nodes = found_node->second;
  nnvm::NodePtr result_node;
  // when unbatchable node was processed, only one entry is added. because it's not NodeEntry, it's
  // not possible to replicate the nodes upfront.
  result_node = new_nodes[entry.index < new_nodes.size() ? entry.index : 0]; // possibly non-batchable multi-out node
  return nnvm::NodeEntry{result_node, 0, entry.version};
}

nnvm::NodePtr MapNode(const nnvm::NodePtr& node,
                      const std::unordered_map<nnvm::Node*,
                                               std::vector<nnvm::NodePtr>>& node_map) {
  nnvm::Node* node_ptr = node.get();
  std::vector<nnvm::NodeEntry>& old_inputs = node_ptr->inputs;
  std::vector<nnvm::NodeEntry> new_inputs;
  new_inputs.reserve(old_inputs.size());
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


struct NodeEntryVectorHash {
  size_t operator()(const std::vector<nnvm::NodeEntry>& nodes) const {
    size_t result = 0;
    auto hash_func = nnvm::NodeEntryHash();
    for (auto n : nodes) {
      result = (result >> 1) ^ hash_func(n);
    }
    return result;
  }
};
struct NodeEntryVectorEqual {
  size_t operator()(const std::vector<nnvm::NodeEntry>& a, const std::vector<nnvm::NodeEntry>& b) const {
    if (a.size() != b.size()) return false;
    auto equal_func = nnvm::NodeEntryEqual();
    for (size_t i = 0; i < a.size(); i++) {
      if (!equal_func(a[i], b[i])) return false;
    }
    return true;
  }
};

typedef typename std::unordered_map<std::vector<nnvm::NodeEntry>, nnvm::NodeEntry,
                                    NodeEntryVectorHash, NodeEntryVectorEqual> ConcatNodeEntryMap;

nnvm::NodeEntry CreateConcatNode(const std::vector<nnvm::NodePtr>& op_ptrs,
                                 uint32_t input_index,
                                 const std::unordered_map<nnvm::Node*, std::vector<nnvm::NodePtr>>& node_map,
                                 const ConcatNodeEntryMap& concat_map) {
  size_t num_nodes = op_ptrs.size();
  size_t in_batch_axis = 0; // TODO get batch axis from nodes for each input index

  // batchable input, create concat node
  std::vector<nnvm::NodeEntry> node_inputs;
  node_inputs.reserve(num_nodes);
  std::string node_name = "batch_concat_in_" + std::to_string(input_index);
  for (const nnvm::NodePtr op_ptr : op_ptrs) {
    nnvm::NodeEntry input_entry = op_ptr.get()->inputs[input_index];
    node_inputs.emplace_back(MapNodeEntry(input_entry, node_map));
    node_name += "_" + input_entry.node->attrs.name;
  }

  auto found = concat_map.find(node_inputs);
  if (found == concat_map.end()) {
    std::unordered_map<std::string, std::string> concat_args = {
      {"num_args", std::to_string(num_nodes)},
      {"dim", std::to_string(in_batch_axis)}
    };

    return MakeNode("concat", node_name, node_inputs, concat_args);
  } else {
    return found->second;
  }
}

nnvm::Graph DBatchEngine::BatchGraphs(const std::vector<nnvm::Graph>& graphs) {
  // collect depth
  static bool collect_debug = dmlc::GetEnv("DB_COLLECT_DEBUG", false);
  int max_depth = 0;
  std::unordered_map<nnvm::Node*, int> depth_map;
  std::vector<nnvm::NodeEntry> prev_outputs, all_old_entries;
  for (const nnvm::Graph& g : graphs) {
    const std::vector<nnvm::NodeEntry>& g_outputs = g.outputs;
    if (collect_debug) LOG(INFO) << "graph outputs " << g_outputs.size();
    std::copy(g_outputs.begin(), g_outputs.end(), std::inserter(prev_outputs, prev_outputs.end()));
    std::copy(g_outputs.begin(), g_outputs.end(), std::inserter(all_old_entries, all_old_entries.end()));
    nnvm::DFSVisit(g_outputs, [&](const nnvm::NodePtr& n){
      int depth = 0;
      for (auto e : n->inputs) {
        all_old_entries.emplace_back(e);
        int idepth = depth_map.at(e.node.get());
        depth = std::max(depth, idepth+1);
      }
      if (collect_debug) LOG(INFO) << "depth of (" << n->attrs.name << "): " << depth;
      depth_map.insert({n.get(), depth});
      max_depth = std::max(depth, max_depth);
    });
  }
  if (collect_debug) {
    LOG(INFO) << "total graph outputs " << prev_outputs.size();
    LOG(INFO) << "max depth " << max_depth;
  }
  // depth: sign->node
  std::vector<std::unordered_map<uint64_t, std::vector<nnvm::NodePtr>>> forward_steps;
  forward_steps.resize(max_depth + 1);

  for (const nnvm::Graph& g : graphs) {
    nnvm::DFSVisit(g.outputs, [&](const nnvm::NodePtr& n){
      uint64_t sign = ComputeNodeSign(n);
      int depth = depth_map.at(n.get());
      forward_steps[depth][sign].emplace_back(n);
    });
  }

  // generate new graph
  static bool rewrite_debug = dmlc::GetEnv("DB_REWRITE_DEBUG", false);
  // create mapping from old node to new node
  std::unordered_map<nnvm::Node*, std::vector<nnvm::NodePtr>> old_new_node_map;
  ConcatNodeEntryMap concat_map;
  for (uint32_t istep = 0; istep < forward_steps.size(); istep++) {

    const std::unordered_map<uint64_t, std::vector<nnvm::NodePtr>>& step = forward_steps[istep];
    for (const std::pair<uint64_t, std::vector<nnvm::NodePtr>> step_ops : step) {

      uint64_t op_sign = step_ops.first;
      const std::vector<nnvm::NodePtr>& op_ptrs = step_ops.second;
      size_t num_nodes = op_ptrs.size();
      nnvm::Node* first_op_node = op_ptrs.front().get();

      if (first_op_node->is_variable()) {
        old_new_node_map[first_op_node].emplace_back(op_ptrs.front());
        continue;
      } else if(num_nodes == 1) { // op that can't be batched, record mapping and move on
        nnvm::NodePtr mapped_node = MapNode(op_ptrs.front(), old_new_node_map);
        old_new_node_map[first_op_node].emplace_back(mapped_node);
        continue;
      }

      size_t num_inputs = first_op_node->num_inputs(), num_outputs = first_op_node->num_outputs();
      if (rewrite_debug) LOG(INFO) << "batchable, nodes: " << num_nodes << ", inputs: " << num_inputs << ", outputs: " << num_outputs;

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
      if (rewrite_debug) {
        LOG(INFO) << "sample lengths: ";
        for (size_t l : sample_lengths) {
          LOG(INFO) << l;
        }
      }

      // concat inputs
      // TODO this can be inefficient. keep a mapping of concat inputs from last step
      std::vector<nnvm::NodeEntry> batch_node_inputs;
      batch_node_inputs.reserve(num_inputs);
      for (uint32_t i = 0; i < num_inputs; i++) {
        if (batchable_data_indices.find(i) != batchable_data_indices.end()) {
          batch_node_inputs.emplace_back(CreateConcatNode(op_ptrs, i, old_new_node_map, concat_map));
        } else { // TODO unbatchable input, signature should not allow different unbatchable inputs
          batch_node_inputs.emplace_back(MapNodeEntry(first_op_node->inputs[i],
                                                      old_new_node_map));
        }
      }
      if (rewrite_debug) LOG(INFO) << "batch node inputs: " << batch_node_inputs.size();

      // create op with concat inputs
      const Op* batch_op = first_op_node->op();
      nnvm::NodeEntry batch_node = MakeNode(batch_op->name.c_str(),
                                            "step_"+std::to_string(istep)+"_batch_"+batch_op->name+"_"+std::to_string(op_sign),
                                            batch_node_inputs, first_op_node->attrs.dict);
      if (rewrite_debug) LOG(INFO) << "created batched op: " << batch_node.node.get()->attrs.name;

      for (uint32_t i = 0; i < num_outputs; i++) {
        nnvm::NodeEntry out_node{batch_node.node, i, batch_node.version};
        std::vector<nnvm::NodeEntry> slice_input = {out_node};
        std::vector<nnvm::NodeEntry> slices;
        slices.reserve(num_nodes);

        // slice according to lengths to get new node
        // TODO this can be inefficient. split only when necessary
        for (uint32_t j = 0, begin = 0; j < num_nodes; j++, begin += sample_lengths[j]) {
          size_t out_batch_axis = 0; // TODO get batch axis for each output
          std::unordered_map<std::string, std::string> slice_args = {
            {"begin", "(" + std::to_string(begin) + ",)"},
            {"end", "(" + std::to_string(begin+sample_lengths[j]) + ",)"}
          };
          nnvm::NodeEntry slice_node = MakeNode("slice", out_node.node.get()->attrs.name+"_slice_"+std::to_string(j),
                                                slice_input, slice_args);
          old_new_node_map[op_ptrs[j].get()].emplace_back(slice_node.node);
          slices.emplace_back(slice_node);
        }
        CHECK(concat_map.find(slices) == concat_map.end()) << "Must not have cycle.";
        concat_map[slices] = out_node;
      }
      if (rewrite_debug) LOG(INFO) << "splitted";
    }
  }
  std::vector<nnvm::NodeEntry> new_outputs;
  new_outputs.reserve(prev_outputs.size());
  for (const nnvm::NodeEntry prev_out : prev_outputs) {
    new_outputs.emplace_back(MapNodeEntry(prev_out, old_new_node_map));
  }
  new_entry_arr_.reserve(all_old_entries.size());
  for (auto e : all_old_entries) {
    auto new_entry = MapNodeEntry(e, old_new_node_map);
    NDArray arr = entry_arr_[e];
    arr.entry_ = new_entry;
    new_entry_arr_[new_entry] = arr;
    if (rewrite_debug) LOG(INFO) << "Recorded " << arr.var() << " in new_entry_arr_";
  }
  nnvm::Graph new_graph;
  new_graph.outputs = new_outputs;
  // Print graph
  if (rewrite_debug) {
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


void DBatchEngine::ExecuteGraph(const nnvm::Graph& fwd_graph) {
  static bool exec_debug = dmlc::GetEnv("DB_EXEC_DEBUG", false);
  if (exec_debug) LOG(INFO) << "ExecuteGraph";
  using namespace nnvm;
  using namespace imperative;
  using AGInfo = Imperative::AGInfo;
  nnvm::Graph graph = fwd_graph;
  auto fwd_outputs = fwd_graph.outputs;

  // g is the forward graph
  size_t num_forward_outputs = graph.outputs.size();

  // Get outputs from the forward graph
  std::vector<NDArray> outputs;
  std::vector<NDArray> ograds;
  std::vector<NodeEntry> ograd_entries;
  outputs.reserve(num_forward_outputs);
  ograds.reserve(num_forward_outputs);
  ograd_entries.reserve(num_forward_outputs);
  for (NodeEntry output_entry : graph.outputs) {
    auto iter = new_entry_arr_.find(output_entry);
    CHECK(iter != new_entry_arr_.end()) << " CANNOT FIND graph.outputs";
    const auto& arr = iter->second;
    if (exec_debug) LOG(INFO) << "graph.outputs var: " << arr.var();
    outputs.push_back(arr);
  }
  // prepare ograds
  for (size_t i = 0; i < outputs.size(); ++i) {
    ograd_entries.emplace_back(NodeEntry{Node::Create(), 0, 0});
    Imperative::AGInfo& info = Imperative::AGInfo::Create(ograd_entries.back().node);
    info.ctx = outputs[i].ctx();
    // TODO(haibin) handle the case where ograd is not 1.0.
    //if (ograds[i] != nullptr) {
      //info.outputs.emplace_back(*ograds[i]);
      //info.outputs.emplace_back(ograds[i]);
    //} else {
      info.outputs.emplace_back(outputs[i].shape(), outputs[i].ctx(),
                                true, outputs[i].dtype());
      info.outputs.back() = static_cast<real_t>(1.0);
    //}
  }

  // Get gradient graph
  if (exec_debug) LOG(INFO) << "Prepare gradient graph";
  Symbol sym;
  sym.outputs = graph.outputs;
  std::vector<NodeEntry> xs;
  std::vector<NDArray*> x_grads;
  std::vector<OpReqType> x_reqs;
  // TODO(haibin) support variables
  {
    std::vector<NodePtr> args = sym.ListInputs(Symbol::kReadOnlyArgs);
    if (exec_debug) LOG(INFO) << "number of args = " << args.size();
    xs.reserve(args.size());
    x_grads.reserve(args.size());
    x_reqs.reserve(args.size());
    for (const auto& i : args) {
      Imperative::AGInfo& info = Imperative::AGInfo::Get(i);
      if (info.grad_req == kNullOp) continue;
      xs.emplace_back(NodeEntry{i, 0, 0});
      x_grads.push_back(&info.out_grads[0]);
      if (exec_debug) LOG(INFO) << "args grad: " << info.out_grads[0].var();
      CHECK(!info.out_grads[0].is_none()) << "None NDArray found for args' grad";
      x_reqs.push_back(info.grad_req);
      info.fresh_out_grad = true;
    }
    CHECK_GT(xs.size(), 0)
        << "There are no inputs in computation graph that require gradients.";
  }

  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
  static const Op* copy_op = Op::Get("_copy");
  Graph g_graph = pass::Gradient(
      graph, graph.outputs, xs, ograd_entries,
      exec::AggregateGradient, nullptr, nullptr,
      zero_ops, "_copy");
  CHECK_EQ(g_graph.outputs.size(), xs.size());
  for (const auto &e : g_graph.outputs) {
    if (e.node->op() == nullptr) {
      if (exec_debug) LOG(INFO) << "null op ptr, create a copy op";
      auto node = Node::Create();
      node->attrs.op = copy_op;
      node->inputs.push_back(e);
      graph.outputs.push_back(NodeEntry{node, 0, 0});
    } else {
      if (exec_debug) LOG(INFO) << "add a backward output to graph";
      graph.outputs.push_back(e);
    }
  }
  // Print graph
  if (exec_debug) {
    Symbol sym;
    sym.outputs = graph.outputs;
    std::cout << "Full New Forward Backward Graph: \n";
    sym.Print(std::cout);
  }

  // prepare execution for fwd bwd graph
  const auto& idx = graph.indexed_graph();
  // get number of nodes used in forward pass
  size_t num_forward_nodes = 0;
  size_t num_forward_entries = 0;
  for (size_t i = 0; i < num_forward_outputs; ++i) {
    num_forward_nodes = std::max(
        num_forward_nodes, static_cast<size_t>(idx.outputs()[i].node_id + 1));
    num_forward_entries = std::max(
        num_forward_entries, static_cast<size_t>(idx.entry_id(idx.outputs()[i])) + 1);
  }
  if (exec_debug) {
    LOG(INFO) << "num_forward_nodes: " << num_forward_nodes;
    LOG(INFO) << "num_forward_entries: " << num_forward_entries;
  }

  // Allocate buffer
  std::vector<NDArray> buff(idx.num_node_entries());
  // reference count for kNullOp, default to 1 to avoid any null op
  // TODO(haibin) initialize with 0 and correctly calculate ref_counts
  std::vector<uint32_t> ref_count(buff.size(), 1);
  // each op has a state
  std::vector<OpStatePtr> states;
  std::vector<NDArray*> arrays;
  arrays.reserve(buff.size());
  for (size_t i = 0; i < buff.size(); ++i) arrays.push_back(&buff[i]);
  // TODO(haibin) support create_graph && retain_graph for 2nd order grads
  const bool create_graph = false;
  const bool retain_graph = false;
  // TODO don't use empty op states
  states.reserve(num_forward_nodes);
  for (size_t i = 0; i < num_forward_nodes; ++i) {
    states.emplace_back(OpStatePtr());
  }
  // forward output ndarray
  for (auto e: fwd_outputs) {
    auto eid = idx.entry_id(e);
    if (new_entry_arr_.find(e) != new_entry_arr_.end()) {
      arrays[eid] = const_cast<NDArray*>(&(new_entry_arr_[e]));
      if (exec_debug) LOG(INFO) << "update arrays[" << eid << "]: " << new_entry_arr_[e].var();
    } else {
      if (exec_debug) LOG(INFO) << "entry id " << eid << " not found in new_entry_arr_";
    }
  }

  nnvm::DFSVisit(graph.outputs, [&](const nnvm::NodePtr& n){
    auto nid = idx.node_id(n.get());
    if (exec_debug) LOG(INFO) << "visit node " << nid;
    for (NodeEntry e : n->inputs) {
      auto eid = idx.entry_id(e);
      if (new_entry_arr_.find(e) != new_entry_arr_.end()) {
        arrays[eid] = const_cast<NDArray*>(&(new_entry_arr_[e]));
        if (exec_debug) LOG(INFO) << "update arrays[" << eid << "]: " << new_entry_arr_[e].var();
      }
    }
  });
  // TODO(haibin) update ref count for fwd graph
  //  states.reserve(num_forward_nodes);
  //  for (size_t i = 0; i < num_forward_nodes; ++i) {
  //    const AGInfo& info = dmlc::get<AGInfo>(idx[i].source->info);
  //    //states.emplace_back(info.state);

  //    for (size_t j = 0; j < info.outputs.size(); ++j) {
  //      size_t eid = idx.entry_id(i, j);
  //      arrays[eid] = const_cast<NDArray*>(&(info.outputs[j]));
  //      if (retain_graph || info.grad_req != kNullOp) ref_count[eid] = 1;
  //    }
  //  }
  //}

  for (size_t i = 0; i < ograd_entries.size(); ++i) {
    if (!idx.exist(ograd_entries[i].node.get())) {
      LOG(INFO) << i << "the ograd entry doesn't exist. continue.";
      continue;
    }
    AGInfo& info = AGInfo::Get(ograd_entries[i].node);
    if (exec_debug) {
      LOG(INFO) << "update arrays[" << idx.entry_id(ograd_entries[i])
                << "] based on ograd_entry " << i << " = " << info.outputs[0].var();
    }
    arrays[idx.entry_id(ograd_entries[i])] = &info.outputs[0];
  }

  for (size_t i = num_forward_outputs; i < graph.outputs.size(); ++i) {
    size_t eid = idx.entry_id(graph.outputs[i]);
    if (exec_debug) {
      LOG(INFO) << "update arrays[" << eid << "] based on backward output "
                << i << " = " << x_grads[i - num_forward_outputs]->var();
    }
    arrays[eid] = x_grads[i - num_forward_outputs];
    ref_count[eid] = 1;
  }

  // Assign context
  auto vctx = PlaceDevice(idx);

  // Infer shape type
  {
    std::pair<uint32_t, uint32_t> node_range, entry_range;
    node_range = {0, idx.num_nodes()};
    entry_range = {0, idx.num_node_entries()};

    ShapeVector shapes;
    shapes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) shapes.emplace_back(i->shape());
    CheckAndInferShape(&graph, std::move(shapes), false,
                       node_range, entry_range);

    DTypeVector dtypes;
    dtypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) dtypes.emplace_back(i->dtype());
    CheckAndInferType(&graph, std::move(dtypes), false,
                      node_range, entry_range);

    StorageTypeVector stypes;
    stypes.reserve(idx.num_node_entries());
    for (const auto& i : arrays) stypes.emplace_back(i->storage_type());
    exec::DevMaskVector dev_mask;
    dev_mask.reserve(idx.num_nodes());
    for (const auto& i : vctx) dev_mask.emplace_back(i.dev_mask());
    CheckAndInferStorageType(&graph, std::move(dev_mask), std::move(stypes), false,
                             node_range, entry_range);
  }

  // TODO(haibin) Calculate ref count
  //for (size_t i = 0; i < idx.num_nodes(); ++i) {
  //  for (const auto& j : idx[i].inputs) {
  //     ++ref_count[idx.entry_id(j)];
  //  }
  //}

  // Assign reqs
  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  // TODO(haibin) update req based on ref_count
  //for (size_t i = num_forward_entries; i < idx.num_node_entries(); ++i) {
  //  if (ref_count[i] == 0) array_reqs[i] = kNullOp;
  //}
  //for (size_t i = num_forward_outputs; i < idx.outputs().size(); ++i) {
  //  size_t eid = idx.entry_id(idx.outputs()[i]);
  //  array_reqs[eid] = x_reqs[i - num_forward_outputs];
  //}

  const auto& shapes = graph.GetAttr<ShapeVector>("shape");
  const auto& dtypes = graph.GetAttr<DTypeVector>("dtype");
  const auto& stypes = graph.GetAttr<StorageTypeVector>("storage_type");
  const auto& dispatch_modes = graph.GetAttr<DispatchModeVector>("dispatch_mode");

  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    auto num_outputs = idx[i].source->num_outputs();
    for (size_t j = 0; j < num_outputs; ++j) {
      auto eid = idx.entry_id(i, j);
      if (!arrays[eid]->is_none()) continue;
      if (exec_debug) LOG(INFO) << "initialize array[" << eid << "]";
      if (stypes[eid] == kDefaultStorage) {
        *arrays[eid] = NDArray(shapes[eid], vctx[i], true, dtypes[eid]);
      } else {
        *arrays[eid] = NDArray(static_cast<NDArrayStorageType>(stypes[eid]),
                               shapes[eid], vctx[i], true, dtypes[eid]);
      }
    }
  }

  // Execution
  bool prev_recording = Imperative::Get()->set_is_recording(create_graph);
  //bool prev_training = Imperative::Get()->set_is_training(is_train);
  //int prev_bulk_size = Engine::Get()->set_bulk_size(backward_bulk_size_);

  Imperative::Get()->RunGraph(retain_graph, idx, arrays, 0, idx.num_nodes(),
           std::move(array_reqs), std::move(ref_count), &states, dispatch_modes);

  Imperative::Get()->set_is_recording(prev_recording);
  //Engine::Get()->set_bulk_size(prev_bulk_size);
  //Imperative::Get()->set_is_training(prev_training);

  // Clear history
  if (!retain_graph) {
    nnvm::DFSVisit(sym.outputs, [&](const nnvm::NodePtr& n) {
      AGInfo::Clear(n);
      n->inputs.clear();
    });
  }
  //if (variables.size()) {
  //  return x_grads;
  //}
  //return {};
}

}  // namespace mxnet
