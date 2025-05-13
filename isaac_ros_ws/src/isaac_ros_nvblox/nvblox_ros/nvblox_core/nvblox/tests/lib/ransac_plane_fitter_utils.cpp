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
#include <random>

#include "nvblox/tests/ransac_plane_fitter_utils.h"

#include "nvblox/tests/utils.h"

namespace nvblox {

namespace test_utils {

std::vector<Vector3f> getPlanePoints(int num_points,
                                     const Vector2f& range_min_max_m,
                                     float z_height) {
  std::vector<Vector3f> plane_points;
  for (int idx = 0; idx < num_points; ++idx) {
    const float random_x =
        test_utils::randomFloatInRange(range_min_max_m[0], range_min_max_m[1]);
    const float random_y =
        test_utils::randomFloatInRange(range_min_max_m[0], range_min_max_m[1]);
    plane_points.emplace_back(Vector3f{random_x, random_y, z_height});
  }
  return plane_points;
}

std::vector<Vector3f> get3DGaussianPoints(int num_points, const Vector3f& mean,
                                          const Vector3f& stddev) {
  std::random_device random_device;
  std::mt19937 random_number_generator(random_device());
  std::normal_distribution<float> distribution_x(mean.x(), stddev.x());
  std::normal_distribution<float> distribution_y(mean.y(), stddev.y());
  std::normal_distribution<float> distribution_z(mean.z(), stddev.z());

  std::vector<Vector3f> point_cloud;
  for (int i = 0; i < num_points; ++i) {
    const float x = distribution_x(random_number_generator);
    const float y = distribution_y(random_number_generator);
    const float z = distribution_z(random_number_generator);
    point_cloud.emplace_back(Vector3f{x, y, z});
  }
  return point_cloud;
}

void verifyPlaneFit(const std::optional<Plane>& maybe_best_plane,
                    const Vector3f& expected_normal, float expected_offset,
                    float epsilon) {
  EXPECT_TRUE(maybe_best_plane);

  const float best_plane_offset = maybe_best_plane->d();
  const Vector3f best_plane_normal = maybe_best_plane->normal();

  const bool is_flipped = (best_plane_offset == -expected_offset);
  if (is_flipped) {
    EXPECT_NEAR(-expected_offset, best_plane_offset, epsilon);
    EXPECT_NEAR((expected_normal - best_plane_normal).norm(), 0, epsilon);
  } else {
    EXPECT_NEAR(expected_offset, best_plane_offset, epsilon);
    EXPECT_NEAR((-expected_normal - best_plane_normal).norm(), 0, epsilon);
  }
}

}  // namespace test_utils

}  // namespace nvblox
