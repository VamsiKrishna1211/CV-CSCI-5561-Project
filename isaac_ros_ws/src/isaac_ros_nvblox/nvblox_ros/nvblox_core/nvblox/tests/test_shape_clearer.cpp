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

#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/shape_clearer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

template <class VoxelType>
class ShapeClearerTest : public ::testing::Test {
 protected:
  static constexpr float kVoxelSizeM{0.2};
  static constexpr float kTruncationDistanceVox{2};
  static constexpr float kTruncationDistanceMeters{kTruncationDistanceVox *
                                                   kVoxelSizeM};

  void SetUp() override {
    // Generate a TSDF layer
    scene_ = test_utils::getSphereInBox();
    scene_.generateLayerFromScene(kTruncationDistanceMeters, &layer_);
    EXPECT_GT(layer_.numAllocatedBlocks(), 0);
  }

  primitives::Scene scene_;
  VoxelBlockLayer<VoxelType> layer_{kVoxelSizeM, MemoryType::kHost};
  EsdfLayer esdf_layer_{kVoxelSizeM, MemoryType::kHost};
  EsdfIntegrator esdf_integrator_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_ = Camera(fu_, fv_, cu_, cv_, width_, height_);
};

using VoxelTypes = ::testing::Types<TsdfVoxel, OccupancyVoxel>;

TYPED_TEST_SUITE(ShapeClearerTest, VoxelTypes);

// Test behaviour of the corner case of empty layer
TYPED_TEST(ShapeClearerTest, EmptyLayer) {
  constexpr float KVoxelSize = 0.05;
  VoxelBlockLayer<TypeParam> layer(KVoxelSize, MemoryType::kHost);

  std::vector<BoundingShape> shape_vec;
  ShapeClearer<VoxelBlockLayer<TypeParam>> clearer;
  const std::vector<Index3D> updated_blocks = clearer.clear(shape_vec, &layer);

  EXPECT_EQ(layer.numAllocatedBlocks(), 0);
  EXPECT_TRUE(updated_blocks.empty());
}

float getVoxelValue(OccupancyVoxel voxel) { return voxel.log_odds; }

float getVoxelValue(TsdfVoxel voxel) { return voxel.weight; }

float isVoxelCleared(OccupancyVoxel voxel) { return voxel.log_odds == 0.f; }

float isVoxelCleared(TsdfVoxel voxel) { return voxel.weight == 0.f; }

template <typename VoxelType>
void testClearingLayer(const VoxelBlockLayer<VoxelType>& layer,
                       const std::vector<BoundingShape>& shape_vec,
                       VoxelBlockLayer<VoxelType>* layer_to_clear) {
  CHECK_NOTNULL(layer_to_clear);
  EXPECT_EQ(layer.size(), layer_to_clear->size());
  EXPECT_GT(layer.size(), 0);
  EXPECT_EQ(layer.voxel_size(), layer_to_clear->voxel_size());

  ShapeClearer<VoxelBlockLayer<VoxelType>> clearer;
  const std::vector<Index3D> updated_blocks =
      clearer.clear(shape_vec, layer_to_clear);

  // Check that clearing the voxels worked as expected.
  int number_of_initial_cleared_values = 0;
  int number_of_updated_cleared_values = 0;
  auto check_clear = [&](const Index3D& block_index, const Index3D& voxel_index,
                         const VoxelType* initial_voxel_ptr) {
    // Get the voxel center.
    const float block_size = layer_to_clear->block_size();
    Vector3f voxel_center = getCenterPositionFromBlockIndexAndVoxelIndex(
        block_size, block_index, voxel_index);

    // Get the updated voxel pointer.
    VoxelType updated_voxel =
        layer_to_clear->getBlockAtIndex(block_index)
            ->voxels[voxel_index(0)][voxel_index(1)][voxel_index(2)];

    if (isVoxelCleared(*initial_voxel_ptr)) {
      number_of_initial_cleared_values++;
    }
    if (isVoxelCleared(updated_voxel)) {
      number_of_updated_cleared_values++;
    }

    // Check if we are inside an AABB.
    bool is_inside_aabb = false;
    for (const auto& shape : shape_vec) {
      if (shape.contains(voxel_center)) {
        is_inside_aabb = true;
      }
    }
    if (is_inside_aabb) {
      // Voxel was cleared.
      EXPECT_TRUE(isVoxelCleared(updated_voxel));
    } else {
      // Voxel was not cleared.
      EXPECT_EQ(getVoxelValue(*initial_voxel_ptr),
                getVoxelValue(updated_voxel));
    }
  };
  callFunctionOnAllVoxels<VoxelType>(layer, check_clear);

  EXPECT_EQ(layer.size(), layer_to_clear->size());  // no deallocation
  EXPECT_GT(updated_blocks.size(), 0);              // we did clear something
  EXPECT_GT(number_of_updated_cleared_values, number_of_initial_cleared_values);
  EXPECT_GT(number_of_updated_cleared_values,
            0);  // we didn't clear everything
}

TYPED_TEST(ShapeClearerTest, TestBoundingBox) {
  // Create the layers.
  VoxelBlockLayer<TypeParam> layer_to_clear(this->kVoxelSizeM,
                                            MemoryType::kHost);
  layer_to_clear.copyFrom(this->layer_);

  // Create the shapes to clear.
  AxisAlignedBoundingBox aabb_1(Vector3f(-2.0, -1.0, -1.0),
                                Vector3f(0.0, 3.0, 2.5));
  AxisAlignedBoundingBox aabb_2(Vector3f(-1.0, -2.0, 2.0),
                                Vector3f(6.0, 1.0, 6.0));
  // AABB 3 is only containing a single voxel center.
  constexpr float HalfVoxelSizeM = this->kVoxelSizeM / 2.0f;
  Vector3f aabb_3_center(2.5, 3.5, 2.1);
  AxisAlignedBoundingBox aabb_3(
      aabb_3_center - HalfVoxelSizeM * Vector3f::Ones(),
      aabb_3_center + HalfVoxelSizeM * Vector3f::Ones());
  std::vector<BoundingShape> shape_vec{aabb_1, aabb_2, aabb_3};

  // Test the clearing.
  testClearingLayer(this->layer_, shape_vec, &layer_to_clear);

  // Additionally test single voxel clearing (subvoxel accuracy).
  const TypeParam* voxel_ptr;
  EXPECT_TRUE(
      getVoxelAtPosition<TypeParam>(layer_to_clear, aabb_3_center, &voxel_ptr));
  EXPECT_TRUE(isVoxelCleared(*voxel_ptr));  // this is inside the aabb_3
  EXPECT_TRUE(getVoxelAtPosition<TypeParam>(
      layer_to_clear, aabb_3_center + this->kVoxelSizeM * Vector3f::Ones(),
      &voxel_ptr));
  EXPECT_FALSE(isVoxelCleared(*voxel_ptr));  // this is outside the aabb_3

  // Store the layers as pointclouds.
  if (FLAGS_nvblox_test_file_output) {
    io::outputVoxelLayerToPly(this->layer_, "original_layer.ply");
    io::outputVoxelLayerToPly(layer_to_clear, "layer_with_cleared_boxes.ply");
    // Also visualize the esdf.
    this->esdf_integrator_.integrateBlocks(layer_to_clear,
                                           layer_to_clear.getAllBlockIndices(),
                                           &this->esdf_layer_);
    io::outputVoxelLayerToPly(this->esdf_layer_,
                              "esdf_layer_with_cleared_boxes.ply");
  }
}

TYPED_TEST(ShapeClearerTest, TestSphere) {
  // Create the layers.
  VoxelBlockLayer<TypeParam> layer_to_clear(this->kVoxelSizeM,
                                            MemoryType::kHost);
  layer_to_clear.copyFrom(this->layer_);

  // Create the shapes to clear.
  BoundingSphere sphere_1(Vector3f(-2.0, 1.0, 1.0), 2.f);
  BoundingSphere sphere_2(Vector3f(0.0, 1.0, 2.0), 3.f);
  // Sphere 3 is only containing a single voxel center.
  Vector3f sphere_3_center(
      -4.1, -4.1, 2.1);  // make sure this coincides with a voxel center
  BoundingSphere sphere_3(sphere_3_center, this->kVoxelSizeM / 2.0f);
  std::vector<BoundingShape> shape_vec{sphere_1, sphere_2, sphere_3};

  // Test the clearing.
  testClearingLayer(this->layer_, shape_vec, &layer_to_clear);

  // Additionally test single voxel clearing (subvoxel accuracy).
  const TypeParam* voxel_ptr;
  EXPECT_TRUE(getVoxelAtPosition<TypeParam>(layer_to_clear, sphere_3_center,
                                            &voxel_ptr));
  EXPECT_TRUE(isVoxelCleared(*voxel_ptr));  // this is inside the sphere_3
  EXPECT_TRUE(getVoxelAtPosition<TypeParam>(
      layer_to_clear, sphere_3_center + this->kVoxelSizeM * Vector3f::Ones(),
      &voxel_ptr));
  EXPECT_FALSE(isVoxelCleared(*voxel_ptr));  // this is outside the sphere_3

  // Store the layers as pointclouds.
  if (FLAGS_nvblox_test_file_output) {
    io::outputVoxelLayerToPly(this->layer_, "original_layer.ply");
    io::outputVoxelLayerToPly(layer_to_clear, "layer_with_cleared_spheres.ply");
    // Also visualize the esdf.
    this->esdf_integrator_.integrateBlocks(layer_to_clear,
                                           layer_to_clear.getAllBlockIndices(),
                                           &this->esdf_layer_);
    io::outputVoxelLayerToPly(this->esdf_layer_,
                              "esdf_layer_with_cleared_spheres.ply");
  }
}

TYPED_TEST(ShapeClearerTest, TestMixedShapes) {
  // Create the layers.
  VoxelBlockLayer<TypeParam> layer_to_clear(this->kVoxelSizeM,
                                            MemoryType::kHost);
  layer_to_clear.copyFrom(this->layer_);

  // Create the shapes to clear.
  BoundingSphere sphere(Vector3f(-2.0, 1.0, 1.0), 2.f);
  AxisAlignedBoundingBox aabb(Vector3f(-1.0, -2.0, 2.0),
                              Vector3f(6.0, 1.0, 6.0));
  std::vector<BoundingShape> shape_vec{sphere, aabb};

  // Test the clearing.
  testClearingLayer(this->layer_, shape_vec, &layer_to_clear);

  // Store the layers as pointclouds.
  if (FLAGS_nvblox_test_file_output) {
    io::outputVoxelLayerToPly(this->layer_, "original_layer.ply");
    io::outputVoxelLayerToPly(layer_to_clear, "layer_with_cleared_shapes.ply");
    // Also visualize the esdf.
    this->esdf_integrator_.integrateBlocks(layer_to_clear,
                                           layer_to_clear.getAllBlockIndices(),
                                           &this->esdf_layer_);
    io::outputVoxelLayerToPly(this->esdf_layer_,
                              "esdf_layer_with_cleared_shapes.ply");
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
