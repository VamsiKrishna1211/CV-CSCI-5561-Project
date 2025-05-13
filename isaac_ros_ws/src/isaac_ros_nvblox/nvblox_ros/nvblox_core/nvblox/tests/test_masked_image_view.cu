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
#include "nvblox/core/unified_ptr.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/sensors/image.h"
#include "nvblox/utils/nvblox_art.h"
namespace nvblox {

namespace {

// Count number of active pixels in mask.
// should be called with a single thread
template <typename MaskedDepthImageViewType>
__global__ void countNumMaskedKernel(const MaskedDepthImageViewType image,
                                     int* result) {
  *result = 0;
  for (int y = 0; y < image.rows(); ++y) {
    for (int x = 0; x < image.cols(); ++x) {
      *result += image.isMasked(y, x);
    }
  }
}
}  // namespace

class MaskedImageViewTestFixture : public ::testing::Test {
 public:
  static constexpr int kRows = 10;
  static constexpr int kCols = 10;

  DepthImage depth_{kRows, kCols, MemoryType::kUnified};
  MonoImage mask_{kRows, kCols, MemoryType::kUnified};
  unified_ptr<int> num_masked_ = make_unified<int>(MemoryType::kUnified);

  template <typename MaskedDepthImageViewType>
  int countNumMasked(const MaskedDepthImageViewType& view) {
    countNumMaskedKernel<<<1, 1, 0, CudaStreamOwning()>>>(view,
                                                          num_masked_.get());
    checkCudaErrors(cudaPeekAtLastError());
    return *num_masked_;
  }

 protected:
  void SetUp() override { *num_masked_ = 0; }
};

TEST_F(MaskedImageViewTestFixture, NoMaskImageDefaultsToActive) {
  EXPECT_EQ(countNumMasked(MaskedDepthImageView(depth_)), kRows * kCols);
  EXPECT_EQ(countNumMasked(MaskedDepthImageConstView(depth_)), kRows * kCols);
}

TEST_F(MaskedImageViewTestFixture, EmptyMaskImage) {
  EXPECT_EQ(countNumMasked(MaskedDepthImageView(depth_, mask_)), 0);
  EXPECT_EQ(countNumMasked(MaskedDepthImageConstView(depth_, mask_)), 0);
}

TEST_F(MaskedImageViewTestFixture, FullMaskImage) {
  std::memset(mask_.dataPtr(), 255, kRows * kCols);
  EXPECT_EQ(countNumMasked(MaskedDepthImageView(depth_, mask_)), kRows * kCols);
  EXPECT_EQ(countNumMasked(MaskedDepthImageConstView(depth_, mask_)),
            kRows * kCols);
}

TEST_F(MaskedImageViewTestFixture, SingleMaskPixel) {
  std::memset(mask_.dataPtr() + (kCols * 3 + 5), 255, 1);
  EXPECT_EQ(countNumMasked(MaskedDepthImageView(depth_, mask_)), 1);
  EXPECT_EQ(countNumMasked(MaskedDepthImageConstView(depth_, mask_)), 1);
}

}  // namespace nvblox

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
