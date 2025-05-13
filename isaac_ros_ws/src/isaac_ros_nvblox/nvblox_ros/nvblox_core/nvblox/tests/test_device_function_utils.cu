/*
Copyright 2024 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <gtest/gtest.h>

#include "nvblox/core/internal/cuda/device_function_utils.cuh"
#include "nvblox/core/unified_ptr.h"

using namespace nvblox;

void __global__ callOnOneThreadKernel(int* int_ptr) {
  callWithFirstThreadInEachBlock([int_ptr]() { atomicAdd(int_ptr, 1); });
}

TEST(DeviceFunctionUtilsTest, CallOnOneThread) {
  // Initialize output variable.
  auto int_ptr = make_unified<int>(MemoryType::kUnified);
  *int_ptr = 0;

  // Call kernel to increment output once per ThreadBlock.
  const int num_thread_blocks = 100;
  const int num_threads_per_block = 128;
  callOnOneThreadKernel<<<num_thread_blocks, num_threads_per_block>>>(
      int_ptr.get());
  cudaDeviceSynchronize();

  // Check that we get exactly one increment per ThreadBlock.
  EXPECT_EQ(num_thread_blocks, *int_ptr);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
