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

#include <cmath>

#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/weighting_function.h"
#include "nvblox/interpolation/interpolation_3d.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/semantics/image_projector.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/projective_tsdf_integrator_cpu.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

class TsdfErrorTest : public ::testing::Test {
 protected:
  TsdfErrorTest() : camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  // Test layer
  constexpr static float kVoxelSizeM = 0.05;
  constexpr static float kBlockSizeM =
      kVoxelSizeM * VoxelBlock<TsdfVoxel>::kVoxelsPerSide;
  DepthImageBackProjector image_back_projector_;

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

TEST_F(TsdfErrorTest, SymmetricViewOnPlane) {
  // The goal of this test is to verify that two symmetric views of a scene lead
  // to two symmetric tsdfs.
  // Setup:
  // - two cameras observing a plane at a slight angle of opposite sign.
  // Test:
  // - the tsdfs coming from both cameras are symmetric
  //   (we verify that the mean of both tsdfs are the same).
  // - the tsdf error is small (compared to the expected tsdf distance).

  // Create scene including a plane.
  constexpr float kPlaneDistance = 3.0f;
  primitives::Scene scene;
  Vector3f normal = Vector3f(0.0, 0.0, -1.0);
  normal.normalize();
  scene.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(0.0, 0.0, kPlaneDistance), normal));

  // Create two symmetric camera poses observing the plane.
  constexpr float kTheta = M_PI / 8.0;
  constexpr float kOffset = 2.0;
  Transform T_S_C1 = Transform::Identity();
  Transform T_S_C2 = Transform::Identity();
  T_S_C1.prerotate(Eigen::AngleAxisf(kTheta, Vector3f::UnitY()));
  T_S_C1.pretranslate(Vector3f(-kOffset, 0.0, 0.0));
  T_S_C2.prerotate(Eigen::AngleAxisf(-kTheta, Vector3f::UnitY()));
  T_S_C2.pretranslate(Vector3f(kOffset, 0.0, 0.0));

  // Generate depth frames from the two camera poses.
  DepthImage depth_frame_1 =
      DepthImage(camera_.height(), camera_.width(), MemoryType::kUnified);
  DepthImage depth_frame_2 =
      DepthImage(camera_.height(), camera_.width(), MemoryType::kUnified);
  constexpr float kMaxDist = 100.0f;
  scene.generateDepthImageFromScene(camera_, T_S_C1, kMaxDist, &depth_frame_1);
  scene.generateDepthImageFromScene(camera_, T_S_C2, kMaxDist, &depth_frame_2);

  // Integrate the depth frames into two separate layers.
  TsdfLayer layer_1(kVoxelSizeM, MemoryType::kUnified);
  TsdfLayer layer_2(kVoxelSizeM, MemoryType::kUnified);
  ProjectiveTsdfIntegrator integrator;
  integrator.integrateFrame(depth_frame_1, T_S_C1, camera_, &layer_1);
  integrator.integrateFrame(depth_frame_2, T_S_C2, camera_, &layer_2);

  // Get back projected 3d points (points on the plane).
  Pointcloud temp_pointcloud(MemoryType::kDevice);
  Pointcloud back_projected_depth_pointcloud_1(MemoryType::kDevice);
  Pointcloud back_projected_depth_pointcloud_2(MemoryType::kDevice);
  image_back_projector_.backProjectOnGPU(depth_frame_1, camera_,
                                         &temp_pointcloud);
  transformPointcloudOnGPU(T_S_C1, temp_pointcloud,
                           &back_projected_depth_pointcloud_1);
  image_back_projector_.backProjectOnGPU(depth_frame_2, camera_,
                                         &temp_pointcloud);
  transformPointcloudOnGPU(T_S_C2, temp_pointcloud,
                           &back_projected_depth_pointcloud_2);

  // Iterate of all voxels of layer_1.
  std::vector<Index3D> block_indices_1 = layer_1.getAllBlockIndices();
  int number_of_checks = 0;
  float total_tsdf_distance_1 = 0;
  float total_tsdf_distance_2 = 0;
  std::vector<Vector3f> checked_points;
  std::vector<float> tsdf_errors;
  checked_points.reserve(block_indices_1.size());
  tsdf_errors.reserve(block_indices_1.size());
  for (size_t i = 0; i < block_indices_1.size(); i++) {
    for (size_t x = 0; x < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; x++) {
      for (size_t y = 0; y < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; y++) {
        for (size_t z = 0; z < VoxelBlock<TsdfVoxel>::kVoxelsPerSide; z++) {
          // Get the indices.
          Index3D block_idx = block_indices_1[i];
          Index3D voxel_idx(x, y, z);

          // Get voxel from layer 1.
          const typename TsdfBlock::ConstPtr block_ptr_1 =
              layer_1.getBlockAtIndex(block_idx);
          if (!block_ptr_1) {
            continue;
          }
          TsdfVoxel tsdf_voxel_1 =
              block_ptr_1->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];
          if (tsdf_voxel_1.weight <= 1e-4 ||
              tsdf_voxel_1.distance >=
                  integrator.get_truncation_distance_m(kVoxelSizeM)) {
            continue;
          }

          // Get voxel from layer 2.
          const typename TsdfBlock::ConstPtr block_ptr_2 =
              layer_2.getBlockAtIndex(block_idx);
          if (!block_ptr_2) {
            continue;
          }
          TsdfVoxel tsdf_voxel_2 =
              block_ptr_2->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];
          if (tsdf_voxel_2.weight <= 1e-4 ||
              tsdf_voxel_2.distance >=
                  integrator.get_truncation_distance_m(kVoxelSizeM)) {
            continue;
          }

          // Accumulate the tsdf distances (for the symmetry test).
          total_tsdf_distance_1 += tsdf_voxel_1.distance;
          total_tsdf_distance_2 += tsdf_voxel_2.distance;

          // Get voxel center.
          Vector3f voxel_center = getCenterPositionFromBlockIndexAndVoxelIndex(
              kBlockSizeM, block_idx, voxel_idx);

          // Calculate the GT tsdf (considering view point dependency of
          // tsdf).
          Vector3f intersect_point;
          float intersect_dist;
          Vector3f camera_to_voxel_direction =
              voxel_center - T_S_C1.translation();
          scene.getRayIntersection(voxel_center, camera_to_voxel_direction,
                                   kMaxDist, &intersect_point, &intersect_dist);
          // The GT tsdf is the depth difference between surface point and
          // voxel center.
          float gt_tsdf_1 =
              (T_S_C1.inverse().rotation() * (intersect_point - voxel_center))
                  .z();

          // Calculate the tsdf error.
          float tsdf_error_1 = gt_tsdf_1 - tsdf_voxel_1.distance;

          // We expect the tsdf error to be small.
          // It is not zero as we do closest point pixel "interpolation".
          EXPECT_LT(tsdf_error_1, kVoxelSizeM / 10.0f);
          number_of_checks++;

          // Visualize the tsdf error.
          if (FLAGS_nvblox_test_file_output) {
            checked_points.push_back(voxel_center);
            tsdf_errors.push_back(tsdf_error_1);
          }
        }
      }
    }
  }
  constexpr int kExpectedNumberOfChecks = 70000;
  EXPECT_GT(number_of_checks, kExpectedNumberOfChecks);

  // We expect the tsdf mean on both layers to be the same because of the
  // symmetric setup.
  float mean_tsdf_distance_1 = total_tsdf_distance_1 / number_of_checks;
  float mean_tsdf_distance_2 = total_tsdf_distance_2 / number_of_checks;
  EXPECT_NEAR(mean_tsdf_distance_1, mean_tsdf_distance_2, 1e-1);

  if (FLAGS_nvblox_test_file_output) {
    // Depth frame
    io::writeToPng("depth_frame_1_tsdf_test.png", depth_frame_1);
    io::writeToPng("depth_frame_2_tsdf_test.png", depth_frame_2);
    io::outputVoxelLayerToPly(layer_1, "tsdf_layer_1.ply");
    io::outputVoxelLayerToPly(layer_2, "tsdf_layer_2.ply");
    // Back projected depth frame
    io::outputPointcloudToPly(back_projected_depth_pointcloud_1,
                              "back_projected_depth_1.ply");
    io::outputPointcloudToPly(back_projected_depth_pointcloud_2,
                              "back_projected_depth_2.ply");
    // All points that got checked.
    io::outputPointsToPly(checked_points, tsdf_errors, "tsdf_errors.ply");
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
