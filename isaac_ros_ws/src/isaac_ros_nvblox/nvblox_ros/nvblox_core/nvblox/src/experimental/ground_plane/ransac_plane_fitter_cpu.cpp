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
#include <random>

#include "nvblox/experimental/ground_plane/ransac_plane_fitter_cpu.h"

namespace nvblox {

RansacPlaneFitterCpu::RansacPlaneFitterCpu()
    : random_number_generator_(random_device_()) {}

std::optional<Plane> RansacPlaneFitterCpu::fit(
    const std::vector<Vector3f>& point_cloud) {
  timing::Timer timer("ground_plane/ransac_plane_fitter_fit_cpu");
  if (point_cloud.size() < 3) {
    return std::nullopt;
  }
  float best_cost = std::numeric_limits<float>::max();
  std::optional<Plane> best_model;
  std::uniform_int_distribution<> distribution(0, point_cloud.size() - 1);

  for (int iteration = 0; iteration < num_ransac_iterations_; ++iteration) {
    // Randomly pick 3 points
    const Vector3f p1 = point_cloud[distribution(random_number_generator_)];
    const Vector3f p2 = point_cloud[distribution(random_number_generator_)];
    const Vector3f p3 = point_cloud[distribution(random_number_generator_)];
    Plane current_plane;
    const auto plane_result =
        Plane::planeFromPoints(p1, p2, p3, &current_plane);
    if (!plane_result) {
      continue;
    }

    const float current_cost =
        computePlaneToPointCloudCost(point_cloud, current_plane);

    // Update the best model if we found a better fit in the current iteration.
    if (current_cost < best_cost) {
      best_cost = current_cost;
      best_model = current_plane;
    }
  }
  timer.Stop();
  return best_model;
}

float RansacPlaneFitterCpu::computePlaneToPointCloudCost(
    const std::vector<Vector3f>& point_cloud, const Plane& plane) {
  switch (ransac_type_) {
    case RansacType::kRansac:
      return computePlaneToPointCloudCostMSAC(point_cloud, plane);
    case RansacType::kMSAC:
      return static_cast<float>(
          computePlaneToPointCloudInlierCostRansac(point_cloud, plane));
    default:
      LOG(FATAL) << "Requested ransac type not implemented";
      return 0.0;
  };
}

int RansacPlaneFitterCpu::computePlaneToPointCloudInlierCostRansac(
    const std::vector<Vector3f>& point_cloud, const Plane& plane) {
  int negative_inliers_count = 0;
  for (const auto& point : point_cloud) {
    if (std::fabs(plane.signedDistance(point)) < ransac_distance_threshold_m_) {
      --negative_inliers_count;
    }
  }
  return negative_inliers_count;
}

float RansacPlaneFitterCpu::computePlaneToPointCloudCostMSAC(
    const std::vector<Vector3f>& point_cloud, const Plane& plane) {
  float cost = 0.0;
  const float squared_distance_threshold_m =
      ransac_distance_threshold_m_ * ransac_distance_threshold_m_;
  for (const auto& point : point_cloud) {
    const float distance_m = std::fabs(plane.signedDistance(point));
    if (distance_m < ransac_distance_threshold_m_) {
      cost += distance_m * distance_m;
    } else {
      cost += squared_distance_threshold_m;
    }
  }
  return cost;
}

void RansacPlaneFitterCpu::ransac_distance_threshold_m(
    float ransac_distance_threshold_m) {
  ransac_distance_threshold_m_ = ransac_distance_threshold_m;
}

float RansacPlaneFitterCpu::ransac_distance_threshold_m() const {
  return ransac_distance_threshold_m_;
}

void RansacPlaneFitterCpu::num_ransac_iterations(int num_ransac_iterations) {
  num_ransac_iterations_ = num_ransac_iterations;
}

int RansacPlaneFitterCpu::num_ransac_iterations() const {
  return num_ransac_iterations_;
}

void RansacPlaneFitterCpu::ransac_type(RansacType ransac_type) {
  ransac_type_ = ransac_type;
}

RansacType RansacPlaneFitterCpu::ransac_type() const { return ransac_type_; }

parameters::ParameterTreeNode RansacPlaneFitterCpu::getParameterTree(
    const std::string& name_remap) const {
  const std::string name =
      (name_remap.empty()) ? "ransac_plane_fitter_cpu" : name_remap;
  return parameters::ParameterTreeNode(
      name, {
                parameters::ParameterTreeNode("ransac_distance_threshold_m:",
                                              ransac_distance_threshold_m_),
                parameters::ParameterTreeNode("num_ransac_iterations:",
                                              num_ransac_iterations_),
                parameters::ParameterTreeNode("ransac_type:",
                                              static_cast<int>(ransac_type_)),
            });
}

}  // namespace nvblox
