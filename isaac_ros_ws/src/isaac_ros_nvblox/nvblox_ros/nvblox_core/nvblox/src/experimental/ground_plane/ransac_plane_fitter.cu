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
limitationj under the License.
*/
#include <assert.h>
#include <cuda/std/limits>

#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/experimental/ground_plane/ransac_plane_fitter.h"
#include "nvblox/geometry/plane.h"

namespace nvblox {

// Seed to initialize the random Kernel states
constexpr int kSeed = 1234;

__global__ void initializeRandomStatesKernel(curandState* random_states,
                                             int num_ransac_iterations,
                                             unsigned long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_ransac_iterations) {
    curand_init(seed, idx, 0, &random_states[idx]);
  }
}

__global__ void ransacKernel(const Vector3f* point_cloud, size_t num_points,
                             int num_ransac_iterations,
                             float ransac_distance_threshold_m,
                             const curandState* random_states,
                             float* costs_global, Plane* planes_global) {
  assert(num_points > 0);
  assert(num_ransac_iterations > 0);
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_ransac_iterations) {
    return;
  }

  curandState local_random_state = random_states[idx];

  // Randomly select points
  const int i1 = curand(&local_random_state) % num_points;
  const int i2 = curand(&local_random_state) % num_points;
  const int i3 = curand(&local_random_state) % num_points;

  const Vector3f p1 = point_cloud[i1];
  const Vector3f p2 = point_cloud[i2];
  const Vector3f p3 = point_cloud[i3];
  Plane current_plane;
  const bool plane_from_points_result =
      Plane::planeFromPoints(p1, p2, p3, &current_plane);
  if (!plane_from_points_result) {
    return;
  }
  // Comput the rescending M-Esimator cost  as described in
  // https://www.robots.ox.ac.uk/~vgg/publications/2000/Torr00/torr00.pdf
  float current_cost = 0.0;
  const float squared_distance_threshold_m =
      ransac_distance_threshold_m * ransac_distance_threshold_m;
  for (int i = 0; i < num_points; ++i) {
    const float distance_m =
        std::fabs(current_plane.signedDistance(point_cloud[i]));
    if (distance_m < ransac_distance_threshold_m) {
      current_cost += distance_m * distance_m;
    } else {
      current_cost += squared_distance_threshold_m;
    }
  }

  costs_global[idx] = current_cost;
  planes_global[idx] = current_plane;
}

RansacPlaneFitter::RansacPlaneFitter(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

std::optional<Plane> RansacPlaneFitter::fit(const Pointcloud& point_cloud) {
  // We need at least three points to form a plane. Exit early.
  if (point_cloud.size() < 3) {
    return std::nullopt;
  }

  // Note: The following resize operations are only called if their respective
  // capacity is smaller then the requested size. No need to additionally check.
  random_states_device_.resizeAsync(num_ransac_iterations_, *cuda_stream_);
  planes_device_.resizeAsync(num_ransac_iterations_, *cuda_stream_);
  costs_device_.resizeAsync(num_ransac_iterations_, *cuda_stream_);

  // Skip if the initial cost has been initialized already.
  if (initial_costs_host_.size() == 0) {
    // Initialize to a large number as we later minimize the cost per iteration.
    // This needs to be only set once, as it's then only used to copy to device.
    initial_costs_host_.resize(num_ransac_iterations_,
                               std::numeric_limits<float>::max());
  }
  costs_device_.copyFromAsync(initial_costs_host_, *cuda_stream_);

  constexpr int threads_per_block = 256;
  const int thread_blocks = num_ransac_iterations_ / threads_per_block + 1;

  // Initialize random states to be used in the ransac cuda kernel
  initializeRandomStatesKernel<<<thread_blocks, threads_per_block, 0,
                                 *cuda_stream_>>>(
      random_states_device_.data(), num_ransac_iterations_, kSeed);
  checkCudaErrors(cudaPeekAtLastError());

  // Run the kernel.
  ransacKernel<<<thread_blocks, threads_per_block, 0, *cuda_stream_>>>(
      point_cloud.dataConstPtr(), point_cloud.size(), num_ransac_iterations_,
      ransac_distance_threshold_m_, random_states_device_.data(),
      costs_device_.data(), planes_device_.data());
  checkCudaErrors(cudaPeekAtLastError());

  // Copy the results back to host.
  costs_host_.copyFromAsync(costs_device_, *cuda_stream_);
  planes_host_.copyFromAsync(planes_device_, *cuda_stream_);
  cuda_stream_->synchronize();

  // Get the lowest cost index.
  auto min_cost_iterator =
      std::min_element(costs_host_.begin(), costs_host_.end());

  // Check if we even found a valid plane.
  const float lowest_cost = *min_cost_iterator;
  if (lowest_cost == std::numeric_limits<float>::max()) {
    return std::nullopt;
  }

  // Get the corresponding best plane.
  const int min_cost_idx =
      std::distance(costs_host_.begin(), min_cost_iterator);
  return planes_host_[min_cost_idx];
}

void RansacPlaneFitter::ransac_distance_threshold_m(
    float ransac_distance_threshold_m) {
  ransac_distance_threshold_m_ = ransac_distance_threshold_m;
}

float RansacPlaneFitter::ransac_distance_threshold_m() const {
  return ransac_distance_threshold_m_;
}

void RansacPlaneFitter::num_ransac_iterations(int num_ransac_iterations) {
  num_ransac_iterations_ = num_ransac_iterations;
}

int RansacPlaneFitter::num_ransac_iterations() const {
  return num_ransac_iterations_;
}

parameters::ParameterTreeNode RansacPlaneFitter::getParameterTree(
    const std::string& name_remap) const {
  const std::string name =
      (name_remap.empty()) ? "ransac_plane_fitter" : name_remap;
  return parameters::ParameterTreeNode(
      name, {
                parameters::ParameterTreeNode("num_ransac_iterations:",
                                              num_ransac_iterations_),
                parameters::ParameterTreeNode("ransac_distance_threshold_m:",
                                              ransac_distance_threshold_m_),
            });
}
}  // namespace nvblox