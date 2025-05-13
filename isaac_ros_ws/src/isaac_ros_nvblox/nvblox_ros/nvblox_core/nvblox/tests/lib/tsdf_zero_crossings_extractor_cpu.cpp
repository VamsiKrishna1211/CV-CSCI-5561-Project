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
#include "nvblox/tests/tsdf_zero_crossings_extractor_cpu.h"
#include "nvblox/core/types.h"

namespace nvblox {

Vector3f TsdfZeroCrossingsExtractorCPU::getZeroCrossing(
    const Index3D& voxel_idx_below, float distance_at_vox_above,
    float distance_at_vox_below, const Index3D& block_idx, float voxel_size_m,
    float block_size_m) {
  // Interpolate between voxel above and voxel below to get zero crossing.
  const Vector3f p_L_below = getCenterPositionFromBlockIndexAndVoxelIndex(
      block_size_m, block_idx, voxel_idx_below);
  CHECK_GT(distance_at_vox_above - distance_at_vox_below, 0.f);
  // Distance up in z from the voxel below position.
  const float distance_m = (-distance_at_vox_below * voxel_size_m) /
                           (distance_at_vox_above - distance_at_vox_below);
  // 3D position of the zero crossing.
  const auto p_L_crossing =
      Vector3f(p_L_below.x(), p_L_below.y(), p_L_below.z() + distance_m);
  return p_L_crossing;
}

bool TsdfZeroCrossingsExtractorCPU::computeZeroCrossingsFromAboveOnCPU(
    const TsdfLayer& tsdf_layer) {
  initializeOutputs();
  // For each block: Cast downwards
  for (const Index3D& tsdf_block_idx : tsdf_layer.getAllBlockIndices()) {
    // Get a block
    TsdfBlock::ConstPtr tsdf_block_ptr =
        tsdf_layer.getBlockAtIndex(tsdf_block_idx);
    CHECK(tsdf_block_ptr);
    // For each block, cast down and look for positive to negative transitions
    for (int x_idx = 0; x_idx < TsdfBlock::kVoxelsPerSide; x_idx++) {
      for (int y_idx = 0; y_idx < TsdfBlock::kVoxelsPerSide; y_idx++) {
        // Cast downwards in this column of voxels.
        VoxelBlockZColumnAccessor<TsdfBlock> column_accessor(x_idx, y_idx,
                                                             tsdf_block_ptr);

        computeZeroCrossingsFromAboveOnCPUInBlock(
            column_accessor, tsdf_block_idx, min_tsdf_weight(),
            tsdf_layer.voxel_size());
      }
    }
  }
  return true;
}

void TsdfZeroCrossingsExtractorCPU::computeZeroCrossingsFromAboveOnCPUInBlock(
    const TsdfVoxelBlockZColumnAccessor& tsdf_column, const Index3D& block_idx,
    const float min_tsdf_weight, const float voxel_size) {
  const float block_size = voxelSizeToBlockSize(voxel_size);
  // Step in z top to bottom
  // Store the preceeding voxel.
  std::optional<TsdfVoxel> last_tsdf_voxel;
  for (int z_idx = TsdfBlock::kVoxelsPerSide - 1; z_idx >= 0; z_idx--) {
    // Grab the current voxel
    const TsdfVoxel& tsdf_voxel = tsdf_column[z_idx];
    // If this voxel has a valid measurement
    if (tsdf_voxel.weight > min_tsdf_weight) {
      // If the preceeding voxel was valid
      if (last_tsdf_voxel.has_value()) {
        CHECK_GT(last_tsdf_voxel.value().weight, min_tsdf_weight);
        // Check if we crossed a zero.
        if (last_tsdf_voxel.value().distance > 0.f &&
            tsdf_voxel.distance <= 0.f) {
          // The distances at this and the preceeding voxel.
          const float distance_at_vox_below = tsdf_voxel.distance;
          const float distance_at_vox_above = (*last_tsdf_voxel).distance;
          const Index3D voxel_idx_below =
              Index3D(tsdf_column.x_idx(), tsdf_column.y_idx(), z_idx);
          CHECK_LT(z_idx + 1, TsdfBlock::kVoxelsPerSide);
          // Use the TSDF to get the crossing at sub-voxel resolution.

          // Ignore what seems to be false positive from gcc:
          // ‘last_tsdf_voxel.nvblox::TsdfVoxel::distance’ may be used
          // uninitialized in this function.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
          const Vector3f p_L_crossing = getZeroCrossing(
              voxel_idx_below, distance_at_vox_above, distance_at_vox_below,
              block_idx, voxel_size, block_size);
#pragma GCC diagnostic pop
          // Add the crossing to the list for output.
          p_L_crossings_vec_.push_back(p_L_crossing);
        }
      }
      // Set this voxel to last voxel
      last_tsdf_voxel = tsdf_voxel;
    } else {
      // This voxel not valid, so not put into last voxel.
      last_tsdf_voxel.reset();
    }
  }
}

std::vector<Vector3f> TsdfZeroCrossingsExtractorCPU::getZeroCrossingsHost() {
  return p_L_crossings_vec_;
}

void TsdfZeroCrossingsExtractorCPU::initializeOutputs() {
  p_L_crossings_vec_.clear();
}

}  // namespace nvblox
