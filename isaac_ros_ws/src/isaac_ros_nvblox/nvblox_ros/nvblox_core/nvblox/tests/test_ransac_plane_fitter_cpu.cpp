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

#include "nvblox/core/types.h"
#include "nvblox/experimental/ground_plane/ground_plane_estimator.h"
#include "nvblox/experimental/ground_plane/ransac_plane_fitter_cpu.h"
#include "nvblox/geometry/plane.h"
#include "nvblox/tests/ransac_plane_fitter_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

namespace {
constexpr float kEps = 1.0e-4;
}  // namespace

class ParameterizedPlaneFitterTest
    : public ::testing::TestWithParam<RansacType> {
 protected:
};

std::string RansacTypeToString(
    const ::testing::TestParamInfo<RansacType>& info) {
  switch (info.param) {
    case RansacType::kRansac:
      return "kRansac";
    case RansacType::kMSAC:
      return "kMSAC";
    default:
      return "Unknown";
  }
}

INSTANTIATE_TEST_CASE_P(ParameterizedPlaneFitterTests,
                        ParameterizedPlaneFitterTest,
                        ::testing::Values(RansacType::kRansac,
                                          RansacType::kMSAC),
                        RansacTypeToString);

TEST_P(ParameterizedPlaneFitterTest, NoPointsPointCloud) {
  const RansacType ransac_type_to_test = GetParam();
  std::vector<Vector3f> empty_point_cloud = {};

  RansacPlaneFitterCpu ransac_plane_fitter_cpu;
  ransac_plane_fitter_cpu.ransac_type(ransac_type_to_test);
  auto maybe_best_plane = ransac_plane_fitter_cpu.fit(empty_point_cloud);
  EXPECT_FALSE(maybe_best_plane);
}

TEST_P(ParameterizedPlaneFitterTest, TwoPointsPointCloud) {
  const RansacType ransac_type_to_test = GetParam();
  std::vector<Vector3f> two_points_point_cloud = {
      Vector3f(0.0f, 0.0f, 0.1f),
      Vector3f(1.0f, 0.0f, 0.1f),
  };

  RansacPlaneFitterCpu ransac_plane_fitter_cpu;
  ransac_plane_fitter_cpu.ransac_type(ransac_type_to_test);
  auto maybe_best_plane = ransac_plane_fitter_cpu.fit(two_points_point_cloud);
  EXPECT_FALSE(maybe_best_plane);
}

TEST_P(ParameterizedPlaneFitterTest, ColliniearPointsPointCloud) {
  const RansacType ransac_type_to_test = GetParam();
  std::vector<Vector3f> collinear_points_point_cloud = {
      Vector3f(0.0f, 0.0f, 0.1f),
      Vector3f(9.0f, 0.0f, 0.1f),
      Vector3f(1.0f, 0.0f, 0.1f),
  };

  RansacPlaneFitterCpu ransac_plane_fitter_cpu;
  ransac_plane_fitter_cpu.ransac_type(ransac_type_to_test);

  auto maybe_best_plane =
      ransac_plane_fitter_cpu.fit(collinear_points_point_cloud);
  EXPECT_FALSE(maybe_best_plane);
}

TEST_P(ParameterizedPlaneFitterTest, FitToKnownPlanarPoints) {
  const RansacType ransac_type_to_test = GetParam();
  // Horizontal plane
  const Vector3f expected_normal(0.0, 0.0, 1.0);
  const float expected_offset = 0.1;
  // Points on that plane
  std::vector<Vector3f> point_cloud = {
      Vector3f(0.0f, 0.0f, 0.1f), Vector3f(1.0f, 0.0f, 0.1f),
      Vector3f(2.0f, 0.0f, 0.1f), Vector3f(0.0f, 1.0f, 0.1f),
      Vector3f(1.0f, 1.0f, 0.1f), Vector3f(2.0f, 1.0f, 0.1f),
      Vector3f(0.5f, 0.5f, 0.1f)};

  RansacPlaneFitterCpu ransac_plane_fitter_cpu;
  ransac_plane_fitter_cpu.ransac_type(ransac_type_to_test);

  auto maybe_best_plane_ransac = ransac_plane_fitter_cpu.fit(point_cloud);
  test_utils::verifyPlaneFit(maybe_best_plane_ransac, expected_normal,
                             expected_offset, kEps);
}

TEST_P(ParameterizedPlaneFitterTest, FitToPlanarPointsWithCorruption) {
  const RansacType ransac_type_to_test = GetParam();
  // Plane point parameters.
  const int num_plane_points = 5000;
  const float plane_height_m = 10.0;
  const float min_range_m = -1000.0;
  const float max_range_m = 1000.0;

  // 3D gaussian noise points parameters.
  const int num_noise_points = 1000;
  const Vector3f noise_points_mean(0.0f, 0.0f, 0.0f);
  const float plane_points_range_m = max_range_m - min_range_m;
  const Vector3f noise_points_stddev(plane_points_range_m, plane_points_range_m,
                                     1.0f);

  // Expected to find a plane & to be a planar plane.
  const Vector3f expected_normal(0.0, 0.0, 1.0);
  const float expected_offset = plane_height_m;

  // Create points on a plane.
  const Vector2f points_range_min_max_m = {min_range_m, max_range_m};
  std::vector<Vector3f> point_cloud = test_utils::getPlanePoints(
      num_plane_points, points_range_min_max_m, plane_height_m);

  // Create noise points.
  const std::vector<Vector3f> noise_point_cloud =
      test_utils::get3DGaussianPoints(num_noise_points, noise_points_mean,
                                      noise_points_stddev);

  // Add the noisy points to the point cloud.
  point_cloud.insert(point_cloud.end(), noise_point_cloud.begin(),
                     noise_point_cloud.end());
  EXPECT_EQ(point_cloud.size(), num_plane_points + num_noise_points);

  // Verify Ransac
  RansacPlaneFitterCpu ransac_plane_fitter_cpu;
  ransac_plane_fitter_cpu.ransac_type(ransac_type_to_test);
  auto maybe_best_plane = ransac_plane_fitter_cpu.fit(point_cloud);
  test_utils::verifyPlaneFit(maybe_best_plane, expected_normal, expected_offset,
                             kEps);
}

TEST(GetPointsWithinMinMaxZTest, ValidPointsWithinRange) {
  const float min_z = 0.0f;
  const float max_z = 6.0f;
  const bool skip_invalid = true;

  const std::vector<Eigen::Vector3f> point_cloud = {
      Eigen::Vector3f(1.0f, 2.0f, 3.0f),           // Valid, within range
      Eigen::Vector3f(4.0f, 5.0f, -1.0f),          // Out of range (below min_z)
      Eigen::Vector3f(6.0f, 7.0f, 10.0f),          // Out of range (above max_z)
      Eigen::Vector3f(8.0f, 9.0f, 5.0f),           // Valid, within range
      Eigen::Vector3f::Zero(),                     // Valid, z=0 (within range)
      Eigen::Vector3f(1.0f, 1.0f, std::nanf("")),  // Invalid (NaN)
      Eigen::Vector3f(1.0f, 1.0f,
                      std::numeric_limits<float>::infinity())  // Invalid (Inf)
  };

  std::vector<Eigen::Vector3f> filtered_points =
      getPointsWithinMinMaxZCPU(point_cloud, min_z, max_z, skip_invalid);

  ASSERT_EQ(filtered_points.size(), 3);
  EXPECT_EQ(filtered_points[0], Eigen::Vector3f(1.0f, 2.0f, 3.0f));
  EXPECT_EQ(filtered_points[1], Eigen::Vector3f(8.0f, 9.0f, 5.0f));
  EXPECT_EQ(filtered_points[2], Eigen::Vector3f::Zero());
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
