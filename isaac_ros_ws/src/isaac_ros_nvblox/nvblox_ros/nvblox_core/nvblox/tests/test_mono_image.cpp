/*
Copyright 2022 NVIDIA CORPORATION

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

#include "nvblox/sensors/image.h"
using namespace nvblox;

class MonoImageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Uninitialized depth frame
    mono_frame_ = MonoImage(rows_, cols_, MemoryType::kUnified);
    mono_frame_.setZeroAsync(CudaStreamOwning());

    // Set a single pixel
    mono_frame_(row_set_, col_set_) = 255;
  }
  int rows_ = 10;
  int cols_ = 10;

  int row_set_ = 6;
  int col_set_ = 8;

  MonoImage mono_frame_{MemoryType::kUnified};
};

TEST_F(MonoImageTest, naiveDownscale) {
  // Downscale the image
  MonoImage downscaled(MemoryType::kUnified);
  image::naiveDownscaleGPUAsync(mono_frame_, 2, &downscaled,
                                CudaStreamOwning());

  EXPECT_EQ(downscaled.rows(), rows_ / 2);
  EXPECT_EQ(downscaled.cols(), cols_ / 2);

  for (int y = 0; y < downscaled.rows(); ++y) {
    for (int x = 0; x < downscaled.cols(); ++x) {
      if (y == row_set_ / 2 && x == col_set_ / 2) {
        EXPECT_EQ(downscaled(y, x), 255);
      } else {
        EXPECT_EQ(downscaled(y, x), 0);
      }
    }
  }
}

TEST_F(MonoImageTest, upscale) {
  // Downscale the image
  MonoImage upscaled(MemoryType::kUnified);
  image::upscaleGPUAsync(mono_frame_, 2, &upscaled, CudaStreamOwning());

  EXPECT_EQ(upscaled.rows(), rows_ * 2);
  EXPECT_EQ(upscaled.cols(), cols_ * 2);

  for (int y = 0; y < upscaled.rows(); ++y) {
    for (int x = 0; x < upscaled.cols(); ++x) {
      if ((y == row_set_ * 2 || y == row_set_ * 2 + 1) &&
          (x == col_set_ * 2 || x == col_set_ * 2 + 1)) {
        EXPECT_EQ(upscaled(y, x), 255);
      } else {
        EXPECT_EQ(upscaled(y, x), 0);
      }
    }
  }
}

TEST_F(MonoImageTest, roundtrip) {
  // Downscale the image
  MonoImage upscaled(MemoryType::kUnified);
  image::upscaleGPUAsync(mono_frame_, 2, &upscaled, CudaStreamOwning());

  MonoImage restored(MemoryType::kUnified);
  image::naiveDownscaleGPUAsync(upscaled, 2, &restored, CudaStreamOwning());

  EXPECT_EQ(restored.rows(), rows_);
  EXPECT_EQ(restored.cols(), cols_);

  for (int y = 0; y < restored.rows(); ++y) {
    for (int x = 0; x < restored.cols(); ++x) {
      if (y == row_set_ && x == col_set_) {
        EXPECT_EQ(restored(y, x), 255);
      } else {
        EXPECT_EQ(restored(y, x), 0);
      }
    }
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
