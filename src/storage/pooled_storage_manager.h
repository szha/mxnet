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
#include <algorithm>
#include <vector>
#include <mutex>
#include <new>
#include "./storage_manager.h"
#include "../common/cuda_utils.h"


namespace mxnet {
namespace storage {

#if MXNET_USE_CUDA
/*!
 * \brief Storage manager with a memory pool on gpu. Memory chunks are reused based on exact size
 * match.
 */
class GPUPooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUPooledStorageManager() {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
    min_chunk_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_MIN_CHUNK", 4096);
    if (min_chunk_ < NDEV) {
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_MIN_CHUNK cannot be set to a value smaller than " << NDEV \
                 << ". Got " << min_chunk_ << ".";
    }
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
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    DirectFreeNoLock(handle);
  }

 private:
  void DirectFreeNoLock(Storage::Handle handle) {
    cudaError_t err = cudaFree(handle.dptr);
    size_t size = std::max(handle.size, min_chunk_);
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
    used_memory_ -= size;
  }

 private:
  void ReleaseAll();
  // used memory
  size_t used_memory_ = 0, min_chunk_;
  // percentage of reserved memory
  int reserve_;
  // number of devices
  const size_t NDEV = 32;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledStorageManager);
};  // class GPUPooledStorageManager

void GPUPooledStorageManager::Alloc(Storage::Handle* handle) {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size_t size = std::max(handle->size, min_chunk_);
  auto&& reuse_it = memory_pool_.find(size);
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
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
    auto&& reuse_pool = reuse_it->second;
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    handle->dptr = ret;
  }
}

void GPUPooledStorageManager::Free(Storage::Handle handle) {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size_t size = std::max(handle.size, min_chunk_);
  auto&& reuse_pool = memory_pool_[size];
  reuse_pool.push_back(handle.dptr);
}

void GPUPooledStorageManager::ReleaseAll() {
  for (auto&& i : memory_pool_) {
    for (auto&& j : i.second) {
      Storage::Handle handle;
      handle.dptr = j;
      handle.size = i.first;
      DirectFreeNoLock(handle);
    }
  }
  memory_pool_.clear();
}

/*!
 * \brief Storage manager with a memory pool, with rounded size, on gpu.
 *
 * This GPU mem pool uses a mixture of nearest pow2 (exponential) rounding and
 * nearest multiple (linear) rounding to help alleviate the memory allocation stress
 * in which the default naive exact-size-match pool falls short, such as in variable-length
 * input/output cases like RNN workloads.
 *
 * \param cutoff the cutoff at which rounding is switched from exponential to linear. It's set
 * through MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF environment variable. Must be between 20 (1 MB)
 * and 34 (16 GB).
 * Suppose the cutoff is X, the memory size buckets look like this:
 * exp2(0), exp2(1), ..., exp2(X), 2*exp2(X), 3*exp2(X), ...
 */
class GPUPooledRoundedStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   */
  GPUPooledRoundedStorageManager() {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
    min_chunk_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_MIN_CHUNK", 4096);
    cut_off_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF", 24);
    if (min_chunk_ < 32) {
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_MIN_CHUNK cannot be set to a value smaller than 32. " \
                 << "Got: " << min_chunk_ << ".";
    }
    if (min_chunk_ != 1ul << log2_round_up(min_chunk_)) {
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_MIN_CHUNK must be a power of 2. Got: " << min_chunk_ << ".";
    } else {
      min_chunk_ = log2_round_up(min_chunk_);
    }
    if (cut_off_ < 20 || cut_off_ > LOG2_MAX_MEM) {
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF cannot be set to a value " \
                 << "smaller than 20 or greater than " << LOG2_MAX_MEM << ". Got: " \
                 << cut_off_ << ".";
    }
    if (cut_off_ < min_chunk_) {
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF cannot be set to a value " \
                 << "smaller than log2 of MXNET_GPU_MEM_POOL_MIN_CHUNK. Got: " \
                 << cut_off_ << " vs " << min_chunk_ << ".";
    }
    memory_pool_ = std::vector<std::vector<void*>>((1ul << (LOG2_MAX_MEM - cut_off_)) + cut_off_);
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledRoundedStorageManager() {
    ReleaseAll();
  }

  void Alloc(Storage::Handle* handle) override;
  void Free(Storage::Handle handle) override;

  void DirectFree(Storage::Handle handle) override {
    handle.size = get_size(get_bucket(handle.size));
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    DirectFreeNoLock(handle);
  }

 private:
#if __SIZEOF_SIZE_T__ == __SIZEOF_LONG__

#if defined(__clang__) || defined(__GNUC__)
#define clz(x) __builtin_clzl(x)
#define ctz(x) __builtin_ctzl(x)

#elif defined(__WINDOWS__)
#define clz(x) __lzcnt64(x)
  uint64_t __inline ctz(uint64_t value) {
    QWORD trailing_zero = 0;
    _BitScanForward64(&trailing_zero, value)
    return trailing_zero;
  }
  uint64_t __inline clz(uint64_t value) {
    QWORD leading_zero = 0;
    _BitScanReverse64(&leading_zero, value)
    return 63 - leading_zero;
  }

#endif  // defined(__clang__) || defined(__GNUC__)

#elif __SIZEOF_SIZE_T__ == __SIZEOF_INT__

#if defined(__clang__) || defined(__GNUC__) || defined(__WINDOWS__)
#define clz(x) __builtin_clz(x)
#define ctz(x) __builtin_ctz(x)

#elif defined(__WINDOWS__)
  uint32_t __inline clz(uint32_t value) {
    DWORD leading_zero = 0;
    _BitScanReverse(&leading_zero, value)
    return 31 - leading_zero;
  }
  uint32_t __inline ctz(uint32_t value) {
    DWORD trailing_zero = 0;
    _BitScanForward(&trailing_zero, value)
    return trailing_zero;
  }

#endif  // defined(__clang__) || defined(__GNUC__)
#endif  // __SIZEOF_SIZE_T__

#if defined(__clang__) || defined(__GNUC__) || defined(__WINDOWS__)
  inline int log2_round_up(size_t s) {
    int result = addr_width - 1 - clz(s);
    return result + ((ctz(s) < result)?1:0);
  }
  inline int div_pow2_round_up(size_t s, int divisor_log2) {
    // (1025, 10) -> 2
    // (2048, 10) -> 2
    // (2049, 10) -> 3
    int ffs = ctz(s);  // find first set
    return (s >> divisor_log2) + (ffs < divisor_log2 ? 1 : 0);
  }
#else
  inline int log2_round_up(size_t s) {
    return static_cast<int>(std::ceil(std::log2(s)));
  }
  inline int div_pow2_round_up(size_t s, int divisor_log2) {
    // (1025, 10) -> 2
    // (2048, 10) -> 2
    // (2049, 10) -> 3
    int divisor = std::pow(2, divisor_log2);
    return s / divisor + (s % divisor ? 1 : 0);
  }
#endif  // defined(__clang__) || defined(__GNUC__) || defined(__WINDOWS__)
  inline int get_bucket(size_t s) {
    int log_size = log2_round_up(s);
    if (log_size > static_cast<int>(cut_off_))
      return div_pow2_round_up(s, cut_off_) - 1 + cut_off_;
    else
      return std::max(log_size, static_cast<int>(min_chunk_));
  }

  inline size_t get_size(int bucket) {
    if (bucket <= static_cast<int>(cut_off_))
      return 1ul << bucket;
    else
      return (bucket - cut_off_ + 1) * (1ul << cut_off_);
  }

  void DirectFreeNoLock(Storage::Handle handle) {
    cudaError_t err = cudaFree(handle.dptr);
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
    used_memory_ -= handle.size;
  }

 private:
  void ReleaseAll();
  // number of devices
  const int NDEV = 32;
  const size_t LOG2_MAX_MEM = 34;
  static const int addr_width = sizeof(size_t) * 8;
  // used memory
  size_t used_memory_ = 0, min_chunk_, cut_off_;
  // percentage of reserved memory
  int reserve_;
  // memory pool
  std::vector<std::vector<void*>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledRoundedStorageManager);
};  // class GPUPooledRoundedStorageManager

void GPUPooledRoundedStorageManager::Alloc(Storage::Handle* handle) {
  int bucket = get_bucket(handle->size);
  size_t size = get_size(bucket);
  auto&& reuse_pool = memory_pool_[bucket];
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

void GPUPooledRoundedStorageManager::Free(Storage::Handle handle) {
  int bucket = get_bucket(handle.size);
  auto&& reuse_pool = memory_pool_[bucket];
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  reuse_pool.push_back(handle.dptr);
}

void GPUPooledRoundedStorageManager::ReleaseAll() {
  for (size_t i = 0; i < memory_pool_.size(); i++) {
    int size = get_size(i);
    for (auto& j : memory_pool_[i]) {
      Storage::Handle handle;
      handle.size = size;
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
