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

/*!
 * Copyright (c) 2015 by Contributors
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA
  #include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA
#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <new>
#include "./storage_manager.h"
#include "../common/cuda_utils.h"


namespace mxnet {
namespace storage {

#if MXNET_USE_CUDA
/*!
 * \brief Storage manager with a memory pool on gpu.
 */
class GPUPooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUPooledStorageManager() {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledStorageManager() {
    ReleaseAll();
  }

  void Alloc(Storage::Handle* handle) override;
  void Free(Storage::Handle handle) override;

  void DirectFree(Storage::Handle handle) override {
    handle.size = 1ul << log2_round_up(handle.size);
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    DirectFreeNoLock(handle);
  }

 private:
  static inline int log2_round_up(size_t s) { // invariant: s > 0
    int fls = __builtin_clzl(s); // find last set
    return std::max(5, (64-fls) + ((__builtin_ctzl(s) < fls - 1)?1:0)); // must be bigger than 32 for nccl scatter
  }
  void DirectFreeNoLock(Storage::Handle handle) {
    cudaError_t err = cudaFree(handle.dptr);
    size_t size = handle.size;
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
    used_memory_ -= size;
  }

 private:
  void ReleaseAll();
  // used memory
  size_t used_memory_ = 0;
  // percentage of reserved memory
  int reserve_;
  // memory pool
  std::array<std::vector<void*>, 64> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledStorageManager);
};  // class GPUPooledStorageManager

void GPUPooledStorageManager::Alloc(Storage::Handle* handle) {
  int log2_size = log2_round_up(handle->size);
  size_t size = 1ul << log2_size;
  auto&& reuse_pool = memory_pool_[log2_size];
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  if (reuse_pool.size() == 0) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free <= total * reserve_ / 100 || size > free - total * reserve_ / 100)
      ReleaseAll();

    void* ret = nullptr;
    cudaError_t e = cudaMalloc(&ret, size);
    if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
      LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
    }
    used_memory_ += size;
    handle->dptr = ret;
  } else {
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    handle->dptr = ret;
  }
}

void GPUPooledStorageManager::Free(Storage::Handle handle) {
  int log2_size = log2_round_up(handle.size);
  auto&& reuse_pool = memory_pool_[log2_size];
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  reuse_pool.push_back(handle.dptr);
}

void GPUPooledStorageManager::ReleaseAll() {
  Storage::Handle handle;
  for (size_t i = 0; i < memory_pool_.size(); i++) {
    handle.size = 1ul << i;
    for (auto& j : memory_pool_[i]) {
      handle.dptr = j;
      DirectFreeNoLock(handle);
    }
    memory_pool_[i].clear();
  }
}

#endif  // MXNET_USE_CUDA

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
