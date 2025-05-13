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
#include <numeric>
#include "nvblox/sensors/image.h"

namespace nvblox {
void fill(DepthImageView view, const float value) {
  for (int y = 0; y < view.rows(); ++y) {
    for (int x = 0; x < view.cols(); ++x) {
      for (int c = 0; c < view.num_elements_per_pixel(); ++c) {
        view(y, x, c) = value;
      }
    }
  }
}

void testCrop(const int num_rows, const int num_cols,
              const int num_elements_per_pixel, const ImageBoundingBox& bbox) {
  constexpr float kExteriorValue = 3.F;
  constexpr float kInteriorValue = 1.F;

  // Create image data with a view. Set all pixels to "exterior"
  std::vector<float> buffer(num_rows * num_cols * num_elements_per_pixel,
                            kExteriorValue);
  DepthImageView view(num_rows, num_cols,
                      sizeof(float) * num_cols * num_elements_per_pixel,
                      num_elements_per_pixel, buffer.data());

  // Crop the view and fill with "interior" pixel value
  const DepthImageView cropped = view.cropped(bbox);
  fill(cropped, kInteriorValue);

  // Buffer stride should not change when cropping
  EXPECT_EQ(cropped.stride_bytes(), view.stride_bytes());
  EXPECT_EQ(cropped.stride_num_elements(), view.stride_num_elements());

  // Check that correct pixels in the original image got modfied
  for (int y = 0; y < view.rows(); ++y) {
    for (int x = 0; x < view.cols(); ++x) {
      for (int c = 0; c < view.num_elements_per_pixel(); ++c) {
        if (!bbox.contains(Index2D{x, y})) {
          EXPECT_EQ(view(y, x, c), kExteriorValue);
        } else {
          EXPECT_EQ(view(y, x, c), kInteriorValue);
        }
      }
    }
  }
}

TEST(ImageViewTest, cropInterior) {
  for (int num_rows = 2; num_rows < 10; ++num_rows) {
    for (int num_cols = 2; num_cols < 10; ++num_cols) {
      for (int num_elements_per_pixel = 1; num_elements_per_pixel < 10;
           ++num_elements_per_pixel) {
        const ImageBoundingBox bbox(Index2D{1, 1},
                                    Index2D{num_cols - 1, num_rows - 1});
        testCrop(num_rows, num_cols, num_elements_per_pixel, bbox);
      }
    }
  }
}

TEST(ImageViewTest, cropSinglePixels) {
  const ImageBoundingBox bbox(Index2D{1, 1}, Index2D{1, 1});
  testCrop(4, 4, 1, bbox);
}

TEST(ImageViewTest, cropColumn) {
  constexpr int kNumRows = 10;
  constexpr int kNumCols = 20;

  for (int col = 0; col < kNumCols; ++col) {
    for (int num_elements_per_pixel = 1; num_elements_per_pixel < 10;
         ++num_elements_per_pixel) {
      const ImageBoundingBox bbox(Index2D{col, 0}, Index2D{col, kNumRows - 1});
      testCrop(kNumRows, kNumCols, num_elements_per_pixel, bbox);
    }
  }
}

TEST(ImageViewTest, cropRow) {
  constexpr int kNumRows = 5;
  constexpr int kNumCols = 7;

  for (int row = 0; row < kNumRows; ++row) {
    for (int num_elements_per_pixel = 1; num_elements_per_pixel < 10;
         ++num_elements_per_pixel) {
      const ImageBoundingBox bbox(Index2D{0, row}, Index2D{kNumCols - 1, row});
      testCrop(kNumRows, kNumCols, num_elements_per_pixel, bbox);
    }
  }
}

TEST(ImageViewTest, wholeImage) {
  constexpr int kNumRows = 5;
  constexpr int kNumCols = 7;

  const ImageBoundingBox bbox(Index2D{0, 0},
                              Index2D{kNumCols - 1, kNumRows - 1});
  testCrop(kNumRows, kNumCols, 3, bbox);
}

TEST(ImageViewtest, multiElement) {
  constexpr int kNumElementsPerPixel = 512;
  constexpr int kRows = 200;
  constexpr int kCols = 300;
  constexpr int kStride = kCols * kNumElementsPerPixel * sizeof(int);

  // Create data with increasing values;
  std::vector<int> buffer(kCols * kRows * kNumElementsPerPixel);
  std::iota(buffer.begin(), buffer.end(), 0);

  ImageView<int> view(kRows, kCols, kStride, kNumElementsPerPixel,
                      buffer.data());

  int linear_idx = 0;
  for (int y = 0; y < kRows; ++y) {
    for (int x = 0; x < kCols; ++x) {
      for (int c = 0; c < kNumElementsPerPixel; ++c) {
        EXPECT_EQ(view(y, x, c), buffer.at(linear_idx));
        ++linear_idx;
      }
    }
  }
}

TEST(ImageViewtest, stridedMultiElement) {
  constexpr int kNumElementsPerPixel = 3;
  constexpr int kRows = 10;
  constexpr int kCols = 11;
  constexpr int kStrideNumElements =
      2 * kCols * kNumElementsPerPixel;  // double stride

  // Create data with increasing values;
  std::vector<int> buffer(kStrideNumElements * kRows);
  std::iota(buffer.begin(), buffer.end(), 0);

  ImageView<int> view(kRows, kCols, kStrideNumElements * sizeof(int),
                      kNumElementsPerPixel, buffer.data());

  int linear_idx = 0;
  for (int y = 0; y < kRows; ++y) {
    for (int x = 0; x < kCols; ++x) {
      if (x == 0) {
        linear_idx = y * kStrideNumElements;
      }
      for (int c = 0; c < kNumElementsPerPixel; ++c) {
        EXPECT_EQ(view(y, x, c), buffer.at(linear_idx));
        ++linear_idx;
      }
    }
  }
}

}  // namespace nvblox

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
