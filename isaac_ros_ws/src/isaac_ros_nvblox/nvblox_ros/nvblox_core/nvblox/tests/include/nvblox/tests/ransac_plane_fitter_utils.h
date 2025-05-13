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
#pragma once

#include "nvblox/geometry/plane.h"

namespace nvblox {
namespace test_utils {

std::vector<Vector3f> getPlanePoints(int num_points,
                                     const Vector2f& range_min_max_m,
                                     float z_height);
std::vector<Vector3f> get3DGaussianPoints(int num_points, const Vector3f& mean,
                                          const Vector3f& stddev);

void verifyPlaneFit(const std::optional<Plane>& maybe_best_plane,
                    const Vector3f& expected_normal, float expected_offset,
                    float epsilon);

}  // namespace test_utils
}  // namespace nvblox