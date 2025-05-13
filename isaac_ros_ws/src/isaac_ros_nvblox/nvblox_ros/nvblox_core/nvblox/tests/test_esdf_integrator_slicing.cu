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
#include <memory>

#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/internal/cuda/esdf_integrator_slicing.cuh"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"

using namespace nvblox;

constexpr bool kOutputDebug = false;

// Just exposes the internal slicing methods for testing.
class TestEsdfIntegrator : public EsdfIntegrator {
 public:
  TestEsdfIntegrator(std::shared_ptr<CudaStream> cuda_stream)
      : EsdfIntegrator(cuda_stream) {}

  using EsdfIntegrator::markSitesInSlice;
};

class EsdfIntegratorSlicingTest : public ::testing::Test {
 protected:
  EsdfIntegratorSlicingTest()
      : cuda_stream_ptr_(std::make_shared<CudaStreamOwning>()),
        esdf_integrator_(cuda_stream_ptr_),
        tsdf_layer_(kVoxelSizeM, MemoryType::kUnified),
        esdf_layer_(kVoxelSizeM, MemoryType::kUnified) {}

  constexpr static float kVoxelSizeM = 0.05;
  constexpr static float kBlockSizeM = voxelSizeToBlockSize(kVoxelSizeM);
  constexpr static float kWeight = 1.0f;
  constexpr static float kFreeDistance = 1.0f;
  constexpr static float kOccupiedDistance = -0.025f;

  std::shared_ptr<CudaStream> cuda_stream_ptr_;
  TestEsdfIntegrator esdf_integrator_;

  TsdfLayer tsdf_layer_;
  EsdfLayer esdf_layer_;

  device_vector<Index3D> updated_blocks_;
  device_vector<Index3D> cleared_blocks_;
};

enum class SliceType { kHeightBased, kPlane };

void setAllTsdfVoxelsInBlock(const float distance, const float weight,
                             TsdfBlock* tsdf_block_ptr) {
  callFunctionOnAllVoxels<TsdfVoxel>(
      tsdf_block_ptr, [&](const Index3D&, TsdfVoxel* tsdf_voxel_ptr) {
        tsdf_voxel_ptr->distance = distance;
        tsdf_voxel_ptr->weight = weight;
      });
}

void expectAllVoxelsOnSliceSites(const EsdfBlock& esdf_block,
                                 const int voxel_z_idx,
                                 const bool expect_site) {
  callFunctionOnAllVoxels<EsdfVoxel>(
      esdf_block,
      [&](const Index3D& voxel_idx, const EsdfVoxel* esdf_voxel_ptr) {
        if (voxel_idx.z() == voxel_z_idx) {
          if (expect_site) {
            EXPECT_TRUE(esdf_voxel_ptr->is_site);
          } else {
            EXPECT_FALSE(esdf_voxel_ptr->is_site);
          }
        }
      });
}

void printSiteForSlice(const EsdfBlock& esdf_block, const int slice_z_idx) {
  for (int x = 0; x < VoxelBlock<bool>::kVoxelsPerSide; x++) {
    for (int y = 0; y < VoxelBlock<bool>::kVoxelsPerSide; y++) {
      std::cout << esdf_block.voxels[x][y][slice_z_idx].is_site << ", ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

class ParameterizedEsdfIntegratorSlicingTest
    : public EsdfIntegratorSlicingTest,
      public ::testing::WithParamInterface<SliceType> {
 protected:
};

TEST_P(ParameterizedEsdfIntegratorSlicingTest, SingleBlock) {
  // Allocate the test blocks
  const Index3D occupied_idx(0, 0, 0);
  const Index3D free_idx(1, 0, 0);
  const Index3D one_site_idx(2, 0, 0);
  const Index3D one_site_just_above_idx(3, 0, 0);
  auto occupied_tsdf_block_ptr = tsdf_layer_.allocateBlockAtIndex(occupied_idx);
  auto free_tsdf_block_ptr = tsdf_layer_.allocateBlockAtIndex(free_idx);
  auto one_site_tsdf_block_ptr = tsdf_layer_.allocateBlockAtIndex(one_site_idx);
  auto one_site_just_above_tsdf_block_ptr =
      tsdf_layer_.allocateBlockAtIndex(one_site_just_above_idx);

  // Set test values
  setAllTsdfVoxelsInBlock(kOccupiedDistance, kWeight,
                          occupied_tsdf_block_ptr.get());
  setAllTsdfVoxelsInBlock(kFreeDistance, kWeight, free_tsdf_block_ptr.get());
  setAllTsdfVoxelsInBlock(kFreeDistance, kWeight,
                          one_site_tsdf_block_ptr.get());
  const Index3D one_site_voxel_idx(3, 3, 3);
  (*one_site_tsdf_block_ptr)(one_site_voxel_idx).distance = kOccupiedDistance;
  setAllTsdfVoxelsInBlock(kFreeDistance, kWeight,
                          one_site_just_above_tsdf_block_ptr.get());
  const Index3D one_site_just_above_voxel_idx(5, 5, 5);
  (*one_site_just_above_tsdf_block_ptr)(one_site_just_above_voxel_idx)
      .distance = kOccupiedDistance;

  // Slice params
  constexpr float kMinSliceHeightM = 0.0f;
  constexpr float kMaxSliceHeightM = 3.5 * kVoxelSizeM;
  constexpr float kOutputSliceHeightM = 0.0f;
  const int output_slice_z_idx = kOutputSliceHeightM / esdf_layer_.voxel_size();
  std::vector<Index3D> blocks_to_update = {occupied_idx, free_idx, one_site_idx,
                                           one_site_just_above_idx};

  // Slice!
  const SliceType slice_type = GetParam();
  if (slice_type == SliceType::kHeightBased) {
    const ConstantZSliceDescription slice_spec{
        .z_min_m = kMinSliceHeightM,
        .z_max_m = kMaxSliceHeightM,
        .z_output_m = kOutputSliceHeightM};
    esdf_integrator_.markSitesInSlice<TsdfLayer>(
        tsdf_layer_, blocks_to_update, slice_spec, nullptr, &esdf_layer_,
        &updated_blocks_, &cleared_blocks_);
  } else if (slice_type == SliceType::kPlane) {
    const Plane horizonal_plane(Vector3f(0.0f, 0.0f, 1.0f),
                                Vector3f(0.0f, 0.0f, kMinSliceHeightM));
    const PlanarSliceDescription slice_spec{
        .ground_plane = horizonal_plane,
        .slice_height_above_plane_m = 0.0f,
        .slice_height_thickness_m = kMaxSliceHeightM - kMinSliceHeightM,
        .z_output_m = kOutputSliceHeightM};
    esdf_integrator_.markSitesInSlice<TsdfLayer>(
        tsdf_layer_, blocks_to_update, slice_spec, nullptr, &esdf_layer_,
        &updated_blocks_, &cleared_blocks_);
  } else {
    CHECK(false) << "Test not implemented for slice type";
  }

  // Test that output ESDF layer has 1 allocated block.
  EXPECT_EQ(esdf_layer_.size(), blocks_to_update.size());
  const auto occupied_esdf_block_ptr =
      esdf_layer_.getBlockAtIndex(occupied_idx);
  const auto free_esdf_block_ptr = esdf_layer_.getBlockAtIndex(free_idx);
  const auto one_site_esdf_block_ptr =
      esdf_layer_.getBlockAtIndex(one_site_idx);
  const auto one_site_just_above_esdf_block_ptr =
      esdf_layer_.getBlockAtIndex(one_site_just_above_idx);
  EXPECT_TRUE(occupied_esdf_block_ptr);
  EXPECT_TRUE(free_esdf_block_ptr);
  EXPECT_TRUE(one_site_esdf_block_ptr);
  EXPECT_TRUE(one_site_just_above_esdf_block_ptr);

  constexpr bool kSite = true;
  constexpr bool kNotSite = false;
  expectAllVoxelsOnSliceSites(*occupied_esdf_block_ptr, output_slice_z_idx,
                              kSite);
  expectAllVoxelsOnSliceSites(*free_esdf_block_ptr, output_slice_z_idx,
                              kNotSite);
  // Tests that we have one site where we expect it.
  callFunctionOnAllVoxels<EsdfVoxel>(
      *one_site_esdf_block_ptr,
      [&](const Index3D& voxel_idx, const EsdfVoxel* esdf_voxel_ptr) {
        const int voxel_idx_z = voxel_idx.z();
        if (voxel_idx_z == output_slice_z_idx) {
          // Expect that theres a site below the one site in this block
          if (voxel_idx.x() == one_site_voxel_idx.x() &&
              voxel_idx.y() == one_site_voxel_idx.y()) {
            EXPECT_TRUE(esdf_voxel_ptr->is_site);
          } else {
            if (esdf_voxel_ptr->is_site) {
              std::cout << voxel_idx.transpose() << std::endl;
            }
            EXPECT_FALSE(esdf_voxel_ptr->is_site);
          }
        }
      });
  expectAllVoxelsOnSliceSites(*one_site_just_above_esdf_block_ptr,
                              output_slice_z_idx, kNotSite);
}

TEST_P(ParameterizedEsdfIntegratorSlicingTest, AcrossBlock) {
  // Allocate the test blocks
  const Index3D lower_idx(0, 0, 0);
  const Index3D upper_idx(0, 0, 1);
  auto lower_tsdf_block_ptr = tsdf_layer_.allocateBlockAtIndex(lower_idx);
  auto upper_tsdf_block_ptr = tsdf_layer_.allocateBlockAtIndex(upper_idx);

  // Set values in test blocks
  setAllTsdfVoxelsInBlock(kFreeDistance, kWeight, lower_tsdf_block_ptr.get());
  setAllTsdfVoxelsInBlock(kFreeDistance, kWeight, upper_tsdf_block_ptr.get());

  // Right now we have two blocks on top of each other. Both totally free.
  // Now lets set one voxel in the top block:
  // - in the slice
  // - just above the slice.
  const Index3D in_voxel_idx(0, 0, 1);
  (*upper_tsdf_block_ptr)(in_voxel_idx).distance = kOccupiedDistance;
  const Index3D out_voxel_idx(1, 1, 2);
  (*upper_tsdf_block_ptr)(in_voxel_idx).distance = kOccupiedDistance;

  // Slice params.
  constexpr float kMinSliceHeightM = 0.0f;
  constexpr float kMaxSliceHeightM = 9.5 * kVoxelSizeM;  // 2nd voxel top block
  constexpr float kOutputSliceHeightM = 0.0f;
  const int output_slice_z_idx = kOutputSliceHeightM / esdf_layer_.voxel_size();

  std::vector<Index3D> blocks_to_update = {lower_idx, upper_idx};

  // Slice!
  const SliceType slice_type = GetParam();
  if (slice_type == SliceType::kHeightBased) {
    const ConstantZSliceDescription slice_spec = {
        .z_min_m = kMinSliceHeightM,
        .z_max_m = kMaxSliceHeightM,
        .z_output_m = kOutputSliceHeightM};
    esdf_integrator_.markSitesInSlice<TsdfLayer>(
        tsdf_layer_, blocks_to_update, slice_spec, nullptr, &esdf_layer_,
        &updated_blocks_, &cleared_blocks_);
  } else if (slice_type == SliceType::kPlane) {
    const Plane horizonal_plane(Vector3f(0.0f, 0.0f, 1.0f),
                                Vector3f(0.0f, 0.0f, kMinSliceHeightM));
    const PlanarSliceDescription slice_spec = {
        .ground_plane = horizonal_plane,
        .slice_height_above_plane_m = 0.0f,
        .slice_height_thickness_m = kMaxSliceHeightM - kMinSliceHeightM,
        .z_output_m = kOutputSliceHeightM};
    esdf_integrator_.markSitesInSlice<TsdfLayer>(
        tsdf_layer_, blocks_to_update, slice_spec, nullptr, &esdf_layer_,
        &updated_blocks_, &cleared_blocks_);
  } else {
    CHECK(false) << "Test not implemented for slice type";
  }

  // Test that output ESDF layer has 1 allocated block.
  EXPECT_EQ(esdf_layer_.size(), 1);
  const auto esdf_block_ptr = esdf_layer_.getBlockAtIndex(lower_idx);
  EXPECT_TRUE(esdf_block_ptr);

  // Tests that we have one site where we expect it.
  callFunctionOnAllVoxels<EsdfVoxel>(
      *esdf_block_ptr,
      [&](const Index3D& voxel_idx, const EsdfVoxel* esdf_voxel_ptr) {
        const int voxel_idx_z = voxel_idx.z();
        if (voxel_idx_z == output_slice_z_idx) {
          // Expect that there's a site below the one site in this block
          if (voxel_idx.x() == in_voxel_idx.x() &&
              voxel_idx.y() == in_voxel_idx.y()) {
            EXPECT_TRUE(esdf_voxel_ptr->is_site);
          } else {
            EXPECT_FALSE(esdf_voxel_ptr->is_site);
          }
        }
      });

  if (kOutputDebug) {
    printSiteForSlice(*esdf_block_ptr, output_slice_z_idx);
  }
}

TEST_F(EsdfIntegratorSlicingTest, PlanarSliceConstructor) {
  // Slice params
  constexpr float kSliceHeightAbovePlaneM = 0.0f;
  constexpr float kSliceHeightThicknessM =
      (VoxelBlock<bool>::kVoxelsPerSide + 1) * kVoxelSizeM;

  // The plane - flat through (0,0,0)
  Plane plane(Vector3f(0.0f, 0.0f, 1.0f), 0.0f);
  PlanarSliceColumnBoundsGetter slicer(plane, kSliceHeightAbovePlaneM,
                                       kSliceHeightThicknessM, kBlockSizeM);

  // We expect that the maximum blocks spanned by a (8+1)*voxel_size width slice
  // is 3.
  EXPECT_EQ(slicer.num_blocks_in_vertical_column(), 3);

  // Flat plane through (0,0,0) - sampled at (0,0), (0,0)
  Index2D block_2d_idx(0, 0);
  Index2D voxel_2d_idx(0, 0);
  ColumnBounds column_bounds =
      slicer.getColumnBounds(block_2d_idx, voxel_2d_idx);
  EXPECT_EQ(column_bounds.min_block_idx_z(), 0);
  EXPECT_EQ(column_bounds.max_block_idx_z(), 1);
  EXPECT_EQ(column_bounds.min_voxel_idx_z(), 0);
  EXPECT_EQ(column_bounds.max_voxel_idx_z(), 1);

  // Flat plane through (0,0,0) - sampled at (1,1), (0,0)
  block_2d_idx = Index2D(1, 1);
  voxel_2d_idx = Index2D(0, 0);
  column_bounds = slicer.getColumnBounds(block_2d_idx, voxel_2d_idx);
  EXPECT_EQ(column_bounds.min_block_idx_z(), 0);
  EXPECT_EQ(column_bounds.max_block_idx_z(), 1);
  EXPECT_EQ(column_bounds.min_voxel_idx_z(), 0);
  EXPECT_EQ(column_bounds.max_voxel_idx_z(), 1);

  // The plane - 45degrees flat through (0,0,1)
  plane = Plane(Vector3f(-1.0f, 0.0f, 1.0f).normalized(),
                Vector3f(0.0f, 0.0f, 1.0f));
  slicer = PlanarSliceColumnBoundsGetter(plane, kSliceHeightAbovePlaneM,
                                         kSliceHeightThicknessM, kBlockSizeM);

  // 45degrees plane through (0,0,1) - sampled at (0,0), (0,0)
  // Bottom: 1m / 0.05 = 20voxels = 2 blocks + 4 voxels
  // Top:    1.45m / 0.05 = 29voxels = 3 blocks + 5 voxels
  block_2d_idx = Index2D(0, 0);
  voxel_2d_idx = Index2D(0, 0);
  column_bounds = slicer.getColumnBounds(block_2d_idx, voxel_2d_idx);
  EXPECT_EQ(column_bounds.min_block_idx_z(), 2);
  EXPECT_EQ(column_bounds.max_block_idx_z(), 3);
  EXPECT_EQ(column_bounds.min_voxel_idx_z(), 4);
  EXPECT_EQ(column_bounds.max_voxel_idx_z(), 5);

  // 45degrees plane through (0,0,1) - sampled at (1,1), (2,2)
  // Bottom: 1.5m / 0.05 = 30voxels = 3 blocks + 6 voxels
  // Top:    1.95m / 0.05 = 39voxels = 4 blocks + 7 voxels
  block_2d_idx = Index2D(1, 1);
  voxel_2d_idx = Index2D(2, 2);
  column_bounds = slicer.getColumnBounds(block_2d_idx, voxel_2d_idx);
  EXPECT_EQ(column_bounds.min_block_idx_z(), 3);
  EXPECT_EQ(column_bounds.max_block_idx_z(), 4);
  EXPECT_EQ(column_bounds.min_voxel_idx_z(), 6);
  EXPECT_EQ(column_bounds.max_voxel_idx_z(), 7);
}

// TEST_F(EsdfIntegratorSlicingTest, 45DegreeSlice) {
TEST_P(ParameterizedEsdfIntegratorSlicingTest, 45DegreeSlice) {
  // NOTE(alexmillane): 45 degree slice through a single block.
  // - Slice thickness 2 voxels.
  // +---+---+---+---+---+---+---+---+
  // |   |   |   |   |   | / |   | / |
  // +---+---+---+---+---+---+---+---+
  // |   |   |   | x | / |   | / |   |
  // +---+---+---+---+---+---+---+---+
  // |   |   |   | / |   | / |   |   |
  // +---+---+---+---+---+---+---+---+
  // |   |   | / |   | / |   |   |   |
  // +---+---+---+---+---+---+---+---+
  // |   | / | x | / |   |   |   |   |
  // +---+---+---+---+---+---+---+---+
  // | / |   | / |   |   |   |   |   |
  // +---+---+---+---+---+---+---+---+
  // |   | /x|   |   |   |   |   |   |
  // +---+---+---+---+---+---+---+---+
  // | / |   |   |   |   |   |   |   |
  // +---+---+---+---+---+---+---+---+

  // Allocate the test blocks
  const Index3D lower_idx(0, 0, 0);
  auto lower_tsdf_block_ptr = tsdf_layer_.allocateBlockAtIndex(lower_idx);

  // Set values in test blocks
  setAllTsdfVoxelsInBlock(kFreeDistance, kWeight, lower_tsdf_block_ptr.get());

  // Right now we have a totally free block
  // Now lets set one voxels
  // - just below the slice.
  // - in the slice.
  // - just above the slice.
  const Index3D in_voxel_1_idx(1, 1, 1);
  (*lower_tsdf_block_ptr)(in_voxel_1_idx).distance = kOccupiedDistance;
  const Index3D in_voxel_2_idx(2, 2, 3);
  (*lower_tsdf_block_ptr)(in_voxel_2_idx).distance = kOccupiedDistance;
  const Index3D out_voxel_idx(3, 3, 6);
  (*lower_tsdf_block_ptr)(out_voxel_idx).distance = kOccupiedDistance;

  // Slice params.
  constexpr float kMinSliceHeightM = 0.0f;
  constexpr float kMaxSliceHeightM = 2.0 * kVoxelSizeM;  // 2nd voxel top block
  constexpr float kOutputSliceHeightM = 0.0f;
  const int output_slice_z_idx = kOutputSliceHeightM / esdf_layer_.voxel_size();

  std::vector<Index3D> blocks_to_update = {lower_idx};

  // Slice!
  const SliceType slice_type = GetParam();
  if (slice_type == SliceType::kHeightBased) {
    const ConstantZSliceDescription slice_spec = {
        .z_min_m = kMinSliceHeightM,
        .z_max_m = kMaxSliceHeightM,
        .z_output_m = kOutputSliceHeightM};
    esdf_integrator_.markSitesInSlice<TsdfLayer>(
        tsdf_layer_, blocks_to_update, slice_spec, nullptr, &esdf_layer_,
        &updated_blocks_, &cleared_blocks_);
  } else if (slice_type == SliceType::kPlane) {
    const Plane fortyfive_plane(Vector3f(-1.0f, 0.0f, 1.0f).normalized(),
                                Vector3f(0.0f, 0.0f, kMinSliceHeightM));
    const PlanarSliceDescription slice_spec = {
        .ground_plane = fortyfive_plane,
        .slice_height_above_plane_m = 0.0f,
        .slice_height_thickness_m = kMaxSliceHeightM - kMinSliceHeightM,
        .z_output_m = kOutputSliceHeightM};
    esdf_integrator_.markSitesInSlice<TsdfLayer>(
        tsdf_layer_, blocks_to_update, slice_spec, nullptr, &esdf_layer_,
        &updated_blocks_, &cleared_blocks_);
  } else {
    CHECK(false) << "Test not implemented for slice type";
  }

  // Test that output ESDF layer has 1 allocated block.
  EXPECT_EQ(esdf_layer_.size(), 1);
  const auto esdf_block_ptr = esdf_layer_.getBlockAtIndex(lower_idx);
  EXPECT_TRUE(esdf_block_ptr);

  // Tests that we have one site where we expect it.
  // - Constant flat slice - one in
  // - 45 degress slice - two in
  if (slice_type == SliceType::kPlane) {
    callFunctionOnAllVoxels<EsdfVoxel>(
        *esdf_block_ptr,
        [&](const Index3D& voxel_idx, const EsdfVoxel* esdf_voxel_ptr) {
          const int voxel_idx_z = voxel_idx.z();
          if (voxel_idx_z == output_slice_z_idx) {
            // Expect that there's a site below the one site in this block
            if (voxel_idx.x() == in_voxel_1_idx.x() &&
                voxel_idx.y() == in_voxel_1_idx.y()) {
              EXPECT_TRUE(esdf_voxel_ptr->is_site);
            } else if (voxel_idx.x() == in_voxel_2_idx.x() &&
                       voxel_idx.y() == in_voxel_2_idx.y()) {
              EXPECT_TRUE(esdf_voxel_ptr->is_site);
            } else if (voxel_idx.x() == out_voxel_idx.x() &&
                       voxel_idx.y() == out_voxel_idx.y()) {
              EXPECT_FALSE(esdf_voxel_ptr->is_site);
            } else {
              EXPECT_FALSE(esdf_voxel_ptr->is_site);
            }
          }
        });
  } else if (slice_type == SliceType::kHeightBased) {
    callFunctionOnAllVoxels<EsdfVoxel>(
        *esdf_block_ptr,
        [&](const Index3D& voxel_idx, const EsdfVoxel* esdf_voxel_ptr) {
          const int voxel_idx_z = voxel_idx.z();
          if (voxel_idx_z == output_slice_z_idx) {
            // Expect that there's a site below the one site in this block
            if (voxel_idx.x() == in_voxel_1_idx.x() &&
                voxel_idx.y() == in_voxel_1_idx.y()) {
              EXPECT_TRUE(esdf_voxel_ptr->is_site);
            } else if (voxel_idx.x() == in_voxel_2_idx.x() &&
                       voxel_idx.y() == in_voxel_2_idx.y()) {
              // NOTE(alexmillane): Now false because it's below the slice
              EXPECT_FALSE(esdf_voxel_ptr->is_site);
            } else if (voxel_idx.x() == out_voxel_idx.x() &&
                       voxel_idx.y() == out_voxel_idx.y()) {
              EXPECT_FALSE(esdf_voxel_ptr->is_site);
            } else {
              EXPECT_FALSE(esdf_voxel_ptr->is_site);
            }
          }
        });
  }

  // Print the costmap
  if (kOutputDebug) {
    printSiteForSlice(*esdf_block_ptr, output_slice_z_idx);
  }
}

TEST_P(ParameterizedEsdfIntegratorSlicingTest, TestScene) {
  // Create a scene with a ground plane and a sphere.
  primitives::Scene scene;
  scene.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, 0.0f),
                                        Vector3f(5.0f, 5.0f, 5.0f));
  scene.addGroundLevel(0.0f);
  scene.addCeiling(5.0f);
  const Vector3f center = Vector3f(0.0f, 0.0f, 2.0f);
  const float radius = 2.0f;
  scene.addPrimitive(std::make_unique<primitives::Sphere>(center, radius));
  // Add bounding planes at 5 meters. Basically makes it sphere in a box.
  scene.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);

  // Get the ground truth SDF for it.
  constexpr float kTruncationDistanceM = 4.0f * kVoxelSizeM;
  scene.generateLayerFromScene(kTruncationDistanceM, &tsdf_layer_);

  // Slice params.
  constexpr float kMinSliceHeightM = 2.0f;
  constexpr float kMaxSliceHeightM = kMinSliceHeightM + 1.0 * kVoxelSizeM;
  constexpr float kOutputSliceHeightM = 0.0f;
  const int output_slice_z_idx = kOutputSliceHeightM / esdf_layer_.voxel_size();

  std::vector<Index3D> blocks_to_update = tsdf_layer_.getAllBlockIndices();

  // Slice!
  const SliceType slice_type = GetParam();
  if (slice_type == SliceType::kHeightBased) {
    const ConstantZSliceDescription slice_spec = {
        .z_min_m = kMinSliceHeightM,
        .z_max_m = kMaxSliceHeightM,
        .z_output_m = kOutputSliceHeightM};
    esdf_integrator_.markSitesInSlice<TsdfLayer>(
        tsdf_layer_, blocks_to_update, slice_spec, nullptr, &esdf_layer_,
        &updated_blocks_, &cleared_blocks_);
  } else if (slice_type == SliceType::kPlane) {
    const Plane fortyfive_plane(Vector3f(-1.0f, 0.0f, 1.0f).normalized(),
                                Vector3f(0.0f, 0.0f, 2.0f));
    const PlanarSliceDescription slice_spec = {
        .ground_plane = fortyfive_plane,
        .slice_height_above_plane_m = 0.0f,
        .slice_height_thickness_m = kMaxSliceHeightM - kMinSliceHeightM,
        .z_output_m = kOutputSliceHeightM};

    esdf_integrator_.markSitesInSlice<TsdfLayer>(
        tsdf_layer_, blocks_to_update, slice_spec, nullptr, &esdf_layer_,
        &updated_blocks_, &cleared_blocks_);
  } else {
    CHECK(false) << "Test not implemented for slice type";
  }

  // Check the points inside are in the sphere.
  // - For the slice at 45degrees, the projection of the slice onto the XY plane
  //   squashes the shape.
  callFunctionOnAllVoxels<EsdfVoxel>(
      esdf_layer_, [&](const Index3D& block_index, const Index3D& voxel_index,
                       const EsdfVoxel* voxel) {
        const Vector3f p = getCenterPositionFromBlockIndexAndVoxelIndex(
            kBlockSizeM, block_index, voxel_index);
        const Vector3f p_from_center = p - center;
        const Vector2f xy_from_center = p_from_center.head<2>();

        if (voxel->is_inside) {
          EXPECT_LT(xy_from_center.norm(), 2.0f);
          if (slice_type == SliceType::kHeightBased) {
            EXPECT_LT(xy_from_center.x(), 2.0f * sqrt(2.0f));
          } else {
            EXPECT_LT(xy_from_center.x(), 2.0f);
          }
        }
      });

  // Debug output
  if (kOutputDebug) {
    // ESDF slice
    bool res = io::outputVoxelLayerToPly(
        esdf_layer_, "esdf_integrator_slicing_test_scene_slice.ply");
    EXPECT_TRUE(res);

    // Mesh
    MeshLayer mesh_layer(kBlockSizeM, MemoryType::kDevice);
    MeshIntegrator mesh_integrator(cuda_stream_ptr_);
    res = mesh_integrator.integrateMeshFromDistanceField(tsdf_layer_,
                                                         &mesh_layer);
    EXPECT_TRUE(res);

    res = io::outputMeshLayerToPly(mesh_layer,
                                   "esdf_integrator_slicing_test_scene.ply");
    EXPECT_TRUE(res);
  }
}

INSTANTIATE_TEST_CASE_P(EsdfIntegratorSlicingTests,
                        ParameterizedEsdfIntegratorSlicingTest,
                        ::testing::Values(SliceType::kHeightBased,
                                          SliceType::kPlane));

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
