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
#include "nvblox/semantics/mask_from_detections.h"

#include "nvblox/core/unified_vector.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

__device__ bool isInsideImage(const int row, const int col, const int num_rows,
                              const int num_cols) {
  return (row >= 0 && row < num_rows && col >= 0 && col < num_cols);
}

// TODO(dtingdahl): Handle dominant modes coming from background
__global__ void createSegmentationMaskKernel(
    const int num_detections, const float mode_proximity_threshold,
    MonoImageView* cropped_mask_views,
    DepthImageConstView* cropped_depth_views) {
  // Number of bins in depth histogram.
  constexpr int kNumBins = 100;
  constexpr float kMinDepth = 1.F;
  constexpr float kMaxDepth = 10.F;
  constexpr float kDepthSpan = kMaxDepth - kMinDepth;

  //----------------------------------
  // INITIALIZE SHARED MEM
  //----------------------------------

  // Each cuda-block will populate it's own histogram, stored in shared memory.
  // It is initialized by the first few threads.
  __shared__ int histogram_sh[kNumBins];
  for (int i = threadIdx.x; i < kNumBins; i += blockDim.x) {
    histogram_sh[i] = 0.f;
  }

  // Initialize scalars
  __shared__ int mode_bin_sh;      // Histogram bin containing the mode
  __shared__ int max_bin_sh;       // Number of elements in the mode bin
  __shared__ float mode_depth_sh;  // Number of elements in the mode bin
  if (threadIdx.x == 0) {
    mode_bin_sh = -1;
    max_bin_sh = -1;
    mode_depth_sh = -1.f;
  }
  __syncthreads();

  //----------------------------------
  // BUILD HISTOGRAMS
  //----------------------------------

  // Each block is responsible for one detection.
  const int detection_idx = blockIdx.x;

  if (detection_idx < num_detections) {
    // Get the view to process
    const DepthImageConstView& depth_image = cropped_depth_views[detection_idx];

    // Process pixels. Each thread will process one or several pixels, depending
    // on the size of the detection bbox.
    for (int i = threadIdx.x; i < depth_image.numel(); i += blockDim.x) {
      // Note: linear image indexing not possible since stride != cols
      const int row = i / depth_image.cols();
      const int col = i % depth_image.cols();
      if (isInsideImage(row, col, depth_image.rows(), depth_image.cols())) {
        const float depth = depth_image(row, col);

        // Add to bin if depth is within valid range.
        if (depth < kMaxDepth && depth > kMinDepth) {
          const int bin = static_cast<int>((kNumBins - 1) *
                                           (depth - kMinDepth) / kDepthSpan);
          assert(bin >= 0);
          assert(bin < kNumBins);

          atomicAdd(&histogram_sh[bin], 1);
        }
      }
    }
  }

  // Wait until the whole histogram has been populated
  __syncthreads();

  //----------------------------------
  // FIND MODE DEPTH
  //----------------------------------

  // Find the num elements of the largest bin
  for (int i = threadIdx.x; i < kNumBins; i += blockDim.x) {
    atomicMax(&max_bin_sh, histogram_sh[i]);
  }
  __syncthreads();

  // Find index of the largest bin
  for (int i = threadIdx.x; i < kNumBins; i += blockDim.x) {
    if (histogram_sh[i] == max_bin_sh) {
      mode_bin_sh = i;
    }
  }
  __syncthreads();

  // Get depth corresponding to the max bin
  if (threadIdx.x == 0) {
    mode_depth_sh =
        static_cast<float>(mode_bin_sh * kDepthSpan) / (kNumBins - 1) +
        kMinDepth;
  }
  __syncthreads();

  //----------------------------------
  // POPULATE SEGMENTATION MASK
  //----------------------------------
  if (detection_idx < num_detections && mode_depth_sh > 0.F) {
    const DepthImageConstView& depth_image = cropped_depth_views[detection_idx];

    for (int i = threadIdx.x; i < depth_image.numel(); i += blockDim.x) {
      const int row = i / depth_image.cols();
      const int col = i % depth_image.cols();
      if (isInsideImage(row, col, depth_image.rows(), depth_image.cols())) {
        const float distance_to_mode =
            fabs(depth_image(row, col) - mode_depth_sh);

        if (distance_to_mode < mode_proximity_threshold) {
          cropped_mask_views[detection_idx](row, col) = image::kMaskedValue;
        }
      }
    }
  }
}

ImageBoundingBox clipToImageBounds(const ImageBoundingBox& detection,
                                   const int rows, const int cols) {
  ImageBoundingBox clipped_detection;
  clipped_detection.min() = detection.min().array().max(0);
  clipped_detection.max() =
      detection.max().array().min(Index2D(cols - 1, rows - 1).array());
  return clipped_detection;
}

void maskFromDetections(const std::vector<ImageBoundingBox>& detections,
                        const DepthImage& depth_image,
                        const float mode_proximity_threshold, MonoImage* mask,
                        const CudaStream& cuda_stream) {
  timing::Timer("image/mask_from_detections");

  // Mask is assumed to be allocated.
  // NOTE: The size of mask is checked during cropping. Basically that the size
  // is sufficient such that all boxes are within the image.
  CHECK_NOTNULL(mask);
  CHECK(mask->rows() > 0 && mask->cols() > 0);

  // NOTE(alexmillane, 2024.09.12): Right now we only support camera setups
  // where the mask and depth images have the same size. In the future we can
  // relax this restriction.
  CHECK_EQ(depth_image.rows(), mask->rows());
  CHECK_EQ(depth_image.cols(), mask->cols());

  // Set the image unmasked to start
  mask->setZeroAsync(cuda_stream);

  // Early exit if no detections.
  if (detections.size() == 0) {
    cuda_stream.synchronize();
    return;
  }

  // Create crops of the depth image using the detection bboxes
  host_vector<MonoImageView> cropped_mask_views;
  host_vector<DepthImageConstView> cropped_depth_views;
  for (const auto& detection : detections) {
    // Clip box to the image size
    auto clipped_detection =
        clipToImageBounds(detection, depth_image.rows(), depth_image.cols());
    // Do crops
    cropped_depth_views.push_back(
        DepthImageConstView(depth_image).cropped(clipped_detection));
    cropped_mask_views.push_back(
        MonoImageView(*mask).cropped(clipped_detection));
  }

  constexpr int kNumThreads = 1024;
  const int num_blocks = detections.size();
  CHECK_GT(num_blocks, 0);
  createSegmentationMaskKernel<<<num_blocks, kNumThreads, 0, cuda_stream>>>(
      detections.size(), mode_proximity_threshold, cropped_mask_views.data(),
      cropped_depth_views.data());
  cuda_stream.synchronize();
}

}  // namespace nvblox
