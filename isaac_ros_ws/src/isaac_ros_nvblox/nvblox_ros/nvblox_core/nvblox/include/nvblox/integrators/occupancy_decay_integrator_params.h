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

#include "nvblox/utils/params.h"

namespace nvblox {
// ======= OCCUPANCY DECAY INTEGRATOR =======
constexpr Param<float>::Description kFreeRegionDecayProbabilityParamDesc{
    "free_region_decay_probability", .55f,
    "The decay probability that is applied to the free region on decay. Must "
    "be in [0.5, 1.0]."};

constexpr Param<float>::Description kOccupiedRegionDecayProbabilityParamDesc{
    "occupied_region_decay_probability", .4f,
    "The decay probability that is applied to the occupied region on decay. "
    "Must be in [0.0, 0.5]."};

constexpr Param<bool>::Description kOccupancyDecayToFreeParamDesc{
    "occupancy_decay_to_free", false,
    "If true we set fully decayed voxels to the free probability. Otherwise "
    "they will set to unkown probability."};

struct OccupancyDecayIntegratorParams {
  Param<float> free_region_decay_probability{
      kFreeRegionDecayProbabilityParamDesc};
  Param<float> occupied_region_decay_probability{
      kOccupiedRegionDecayProbabilityParamDesc};
  Param<bool> occupancy_decay_to_free{kOccupancyDecayToFreeParamDesc};
};

}  // namespace nvblox
