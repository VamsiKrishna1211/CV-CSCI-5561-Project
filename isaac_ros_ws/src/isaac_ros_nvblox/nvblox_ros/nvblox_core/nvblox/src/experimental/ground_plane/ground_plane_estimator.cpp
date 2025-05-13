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

#include "nvblox/experimental/ground_plane/ground_plane_estimator.h"

namespace nvblox {

GroundPlaneEstimator::GroundPlaneEstimator(
    std::shared_ptr<CudaStream> cuda_stream)
    : tsdf_zero_crossings_extractor_(cuda_stream),
      ransac_plane_fitter_(cuda_stream),
      cuda_stream_(cuda_stream) {}

std::optional<Plane> GroundPlaneEstimator::computeGroundPlane(
    const TsdfLayer& tsdf_layer) {
  if (std::optional<std::vector<Vector3f>> maybe_tsdf_zero_crossings =
          tsdf_zero_crossings_extractor_.computeZeroCrossingsFromAboveOnGPU(
              tsdf_layer)) {
    tsdf_zero_crossings_ = maybe_tsdf_zero_crossings;
  } else {
    resetInternal();
    return std::nullopt;
  }

  // Filter points outside of the specified range.
  tsdf_zero_crossings_ground_candidates_ = getPointsWithinMinMaxZCPU(
      tsdf_zero_crossings_.value(), ground_points_candidates_min_z_m_,
      ground_points_candidates_max_z_m_);

  // Copy to a device pointcloud
  tsdf_zero_crossings_ground_candidates_point_cloud_.copyFromAsync(
      tsdf_zero_crossings_ground_candidates_.value(), *cuda_stream_);
  cuda_stream_->synchronize();

  if (std::optional<Plane> maybe_ground_plane = ransac_plane_fitter_.fit(
          tsdf_zero_crossings_ground_candidates_point_cloud_)) {
    ground_plane_ = maybe_ground_plane;
  } else {
    resetInternal();
    return std::nullopt;
  }
  return ground_plane_;
}

void GroundPlaneEstimator::resetInternal() {
  tsdf_zero_crossings_ = std::nullopt;
  tsdf_zero_crossings_ground_candidates_ = std::nullopt;
  ground_plane_ = std::nullopt;
}

std::optional<std::vector<Vector3f>> GroundPlaneEstimator::tsdf_zero_crossings()
    const {
  return tsdf_zero_crossings_;
};

std::optional<std::vector<Vector3f>>
GroundPlaneEstimator::tsdf_zero_crossings_ground_candidates() const {
  return tsdf_zero_crossings_ground_candidates_;
}

std::optional<Plane> GroundPlaneEstimator::ground_plane() const {
  return ground_plane_;
}

float GroundPlaneEstimator::ground_points_candidates_min_z_m() const {
  return ground_points_candidates_min_z_m_;
}

float GroundPlaneEstimator::ground_points_candidates_max_z_m() const {
  return ground_points_candidates_max_z_m_;
}

void GroundPlaneEstimator::ground_points_candidates_min_z_m(
    float ground_points_candidates_min_z_m) {
  ground_points_candidates_min_z_m_ = ground_points_candidates_min_z_m;
}

void GroundPlaneEstimator::ground_points_candidates_max_z_m(
    float ground_points_candidates_max_z_m) {
  ground_points_candidates_max_z_m_ = ground_points_candidates_max_z_m;
}

parameters::ParameterTreeNode GroundPlaneEstimator::getParameterTree(
    const std::string& name_remap) const {
  const std::string name =
      (name_remap.empty()) ? "ground_plane_estimator" : name_remap;
  return parameters::ParameterTreeNode(
      name, {parameters::ParameterTreeNode("ground_points_candidates_min_z_m:",
                                           ground_points_candidates_min_z_m_),
             parameters::ParameterTreeNode("ground_points_candidates_max_z_m:",
                                           ground_points_candidates_max_z_m_),
             ransac_plane_fitter_.getParameterTree(),
             tsdf_zero_crossings_extractor_.getParameterTree()});
}

std::vector<Vector3f> getPointsWithinMinMaxZCPU(
    const std::vector<Vector3f>& point_cloud, float min_z, float max_z,
    bool skip_invalid) {
  timing::Timer timer("ground_plane/get_points_within_min_max_z_cpu");
  std::vector<Vector3f> point_cloud_filtered;
  // Only get the points within z range and skip invalid values (nan & inf)
  for (const Vector3f& point : point_cloud) {
    if (skip_invalid && !point.allFinite()) {
      continue;
    }
    if (point.z() >= min_z && point.z() <= max_z) {
      point_cloud_filtered.emplace_back(point);
    }
  }
  timer.Stop();
  return point_cloud_filtered;
}

}  // namespace nvblox
