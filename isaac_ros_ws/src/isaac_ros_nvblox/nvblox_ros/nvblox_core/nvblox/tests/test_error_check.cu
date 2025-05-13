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

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/internal/error_check.h"

using namespace nvblox;

// Test kernel that calls a given functor
template <class Functor>
__global__ void testKernel(const Functor* f) {
  (*f)();
}

// Run the test kernel and check for success
template <class Functor>
void runTestKernel(Functor f, const bool expected_success) {
  cudaDeviceReset();
  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);

  testKernel<<<1, 1, 0, cuda_stream>>>(&f);
  checkCudaErrors(cudaPeekAtLastError());

  const cudaError_t result = cudaStreamSynchronize(cuda_stream);
  if (expected_success) {
    EXPECT_EQ(result, cudaSuccess);
  } else {
    EXPECT_NE(result, cudaSuccess);
  }
}

// Test passing "true"
struct CheckPass {
  __device__ void operator()() const {
    NVBLOX_CHECK(true, "A message");
    ;
  }
};
TEST(ErrorCheck, checkPass) { runTestKernel(CheckPass(), true); }

// Check that the macro can handle assignment inside the assertion clause
// without breaking.
struct CheckPassAssignment {
  __device__ void operator()() const {
    int a;
    NVBLOX_CHECK(a = 1, "A message");
    static_cast<void>(a);
  }
};
TEST(ErrorCheck, CheckPassAssignment) {
  runTestKernel(CheckPassAssignment(), true);
}

// Check that conditional clause without brackets (normally a bad idea) can be
// handled.
struct CheckPassConditionWithoutBrackets {
  __device__ bool operator()() const {
    if (true)
      return 1;
    else
      return 0;
  }
};
TEST(ErrorCheck, CheckPassConditionWithoutBrackets) {
  runTestKernel(CheckPassConditionWithoutBrackets(), true);
}

// IMPORTANT: This failing test must be executed last, since any CUDA calls
// beyond this will keep on failing.
struct CheckFail {
  __device__ bool operator()() const {
    NVBLOX_CHECK(false, "A message");
    return false;
  }
};
TEST(ErrorCheck, checkFail) { runTestKernel(CheckFail(), false); }

TEST(ErrorCheck, checkSuccessHost) { NVBLOX_CHECK(true, ""); }

constexpr const char* kExpectedMessage = "griseknoen";
void checkFail() { NVBLOX_CHECK(false, kExpectedMessage); }
TEST(ErrorCheck, checFailkHost) { ASSERT_DEATH(checkFail(), kExpectedMessage); }

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
