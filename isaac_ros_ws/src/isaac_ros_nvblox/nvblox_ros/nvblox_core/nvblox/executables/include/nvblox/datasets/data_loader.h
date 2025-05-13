/*
Copyright 2022 NVIDIA CORPORATION

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

#include "nvblox/core/types.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace datasets {

enum class DataLoadResult { kSuccess, kBadFrame, kNoMoreData };

class RgbdDataLoaderInterface {
 public:
  RgbdDataLoaderInterface() = default;
  virtual ~RgbdDataLoaderInterface() = default;

  /// Interface for a function that loads the next frames in a dataset
  /// This version of the function should be used when the color and depth
  /// camera are the same.
  ///@param[out] depth_frame_ptr The loaded depth frame.
  ///@param[out] T_L_C_ptr Transform from Camera to the Layer frame.
  ///@param[out] camera_ptr The intrinsic camera model.
  ///@param[out] color_frame_ptr Optional, load color frame.
  ///@return Whether loading succeeded.
  virtual DataLoadResult loadNext(DepthImage* depth_frame_ptr,  // NOLINT
                                  Transform* T_L_C_ptr,         // NOLINT
                                  Camera* camera_ptr,           // NOLINT
                                  ColorImage* color_frame_ptr = nullptr) = 0;

  /// Interface for a function that loads the next frames in a dataset.
  /// This is the version of the function for different depth and color cameras.
  ///@param[out] depth_frame_ptr The loaded depth frame.
  ///@param[out] T_L_D_ptr Transform from depth camera to the Layer frame.
  ///@param[out] depth_camera_ptr The intrinsic depth camera model.
  ///@param[out] color_frame_ptr The loaded color frame.
  ///@param[out] T_L_C_ptr Transform from color camera to the Layer frame.
  ///@param[out] color_camera_ptr The intrinsic color camera model.
  ///@return Whether loading succeeded.
  virtual DataLoadResult loadNext(DepthImage* depth_frame_ptr,  // NOLINT
                                  Transform* T_L_D_ptr,         // NOLINT
                                  Camera* depth_camera_ptr,     // NOLINT
                                  ColorImage* color_frame_ptr,  // NOLINT
                                  Transform* T_L_C_ptr,         // NOLINT
                                  Camera* color_camera_ptr) = 0;

  /// Indicates if the data loader was successfully set up.
  /// @return True if the DataLoader was successfully set up.
  bool setup_success() const { return setup_success_; }

 protected:
  // Indicates if the dataset loader was constructed in a state that was good to
  // go. Initializes to true, so child class constructors indicate failure by
  // setting it to false;
  bool setup_success_ = true;
};

}  // namespace datasets
}  // namespace nvblox