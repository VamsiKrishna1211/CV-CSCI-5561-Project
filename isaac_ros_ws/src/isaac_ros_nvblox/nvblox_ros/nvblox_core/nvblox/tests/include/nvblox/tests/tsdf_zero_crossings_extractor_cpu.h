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
#include <glog/logging.h>

#include "nvblox/core/types.h"
#include "nvblox/experimental/ground_plane/tsdf_zero_crossings_extractor.h"

namespace nvblox {

template <typename VoxelBlockType>
class VoxelBlockZColumnAccessor {
 public:
  using VoxelType = typename VoxelBlockType::VoxelType;

  VoxelBlockZColumnAccessor(const int x_idx, const int y_idx,
                            typename VoxelBlockType::ConstPtr block_ptr)
      : block_ptr_(block_ptr), x_idx_(x_idx), y_idx_(y_idx) {}

  const VoxelType& operator[](const int z_idx) const {
    return block_ptr_->voxels[x_idx_][y_idx_][z_idx];
  }

  int x_idx() const { return x_idx_; }
  int y_idx() const { return y_idx_; }

 private:
  typename VoxelBlockType::ConstPtr block_ptr_;
  const int x_idx_;
  const int y_idx_;
};
using TsdfVoxelBlockZColumnAccessor = VoxelBlockZColumnAccessor<TsdfBlock>;

class TsdfZeroCrossingsExtractorCPU : public TsdfZeroCrossingsExtractor {
 public:
  TsdfZeroCrossingsExtractorCPU() : TsdfZeroCrossingsExtractor(nullptr) {}
  virtual ~TsdfZeroCrossingsExtractorCPU() {}

  bool computeZeroCrossingsFromAboveOnCPU(const TsdfLayer& tsdf_layer);

  std::vector<Vector3f> getZeroCrossingsHost();

 protected:
  void initializeOutputs();

  void computeZeroCrossingsFromAboveOnCPUInBlock(
      const TsdfVoxelBlockZColumnAccessor& tsdf_column,
      const Index3D& block_idx, const float min_tsdf_weight,
      const float voxel_size);

  Vector3f getZeroCrossing(const Index3D& voxel_idx_below,
                           float distance_at_vox_above,
                           float distance_at_vox_below,
                           const Index3D& block_idx, float voxel_size_m,
                           float block_size_m);

  std::vector<Vector3f> p_L_crossings_vec_;
};

}  // namespace nvblox
