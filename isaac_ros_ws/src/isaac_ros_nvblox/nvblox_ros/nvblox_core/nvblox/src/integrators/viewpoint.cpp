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
#include "nvblox/integrators/viewpoint.h"

namespace nvblox {

ViewBasedInclusionData::ViewBasedInclusionData(
    Transform _T_L_C, Camera _camera,
    std::optional<const DepthImage*> _depth_image,
    std::optional<float> _max_view_distance_m,
    std::optional<float> _truncation_distance_m)
    : T_L_C(_T_L_C),
      camera(_camera),
      depth_image(_depth_image),
      max_view_distance_m(_max_view_distance_m),
      truncation_distance_m(_truncation_distance_m) {}

}  // namespace nvblox
