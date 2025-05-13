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
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/map/common_names.h"
#include "nvblox/serialization/layer_cake_streamer.h"

using namespace nvblox;

TEST(LayerCakeStreamer, AddLayerTypes) {
  // Add and get some streamers
  LayerCakeStreamer layer_cake_streamer;
  layer_cake_streamer.add<TsdfLayer>();
  const TsdfLayerStreamerOldestBlocks* tsdf_layer_ptr =
      layer_cake_streamer.getConstPtr<TsdfLayer>();
  EXPECT_NE(tsdf_layer_ptr, nullptr);
  layer_cake_streamer.add<MeshLayer>();
  MeshLayerStreamerOldestBlocks* mesh_layer_ptr =
      layer_cake_streamer.getPtr<MeshLayer>();
  EXPECT_NE(mesh_layer_ptr, nullptr);

  // Try to get a layer streamer that hasn't been added
  const ColorLayerStreamerOldestBlocks* color_layer_ptr =
      layer_cake_streamer.getConstPtr<ColorLayer>();
  EXPECT_EQ(color_layer_ptr, nullptr);

  // Try the get() function which check-fails if the layer streamer isn't
  // present. Nothing to check on the output, but if this is broken the test
  // will crash, and therefore fail.
  layer_cake_streamer.add<ColorLayer>();
  const auto& color_layer_streamer = layer_cake_streamer.get<ColorLayer>();
  (void)color_layer_streamer;
}

TEST(LayerCakeStreamer, CreateFactory) {
  auto layer_cake_streamer =
      LayerCakeStreamer::create<TsdfLayer, ColorLayer, MeshLayer>();
  EXPECT_NE(layer_cake_streamer.getConstPtr<TsdfLayer>(), nullptr);
  EXPECT_NE(layer_cake_streamer.getConstPtr<ColorLayer>(), nullptr);
  EXPECT_NE(layer_cake_streamer.getConstPtr<MeshLayer>(), nullptr);
}

TEST(LayerCakeStreamer, Empty) {
  LayerCakeStreamer layer_cake_streamer;
  auto streamer_ptr = layer_cake_streamer.getConstPtr<TsdfLayer>();
  EXPECT_EQ(streamer_ptr, nullptr);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
