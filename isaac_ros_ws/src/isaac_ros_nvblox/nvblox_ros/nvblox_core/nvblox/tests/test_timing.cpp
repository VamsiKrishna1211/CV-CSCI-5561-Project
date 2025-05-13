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
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/utils/timing.h"

using namespace nvblox;

TEST(TimingTest, TestAccumulatorDouble) {
  timing::Accumulator<double, double, 4> acc;

  for (int i = 0; i < 5; ++i) {
    double value = static_cast<double>(i);
    acc.Add(value);
  }

  EXPECT_EQ(acc.TotalSamples(), 5);
  EXPECT_EQ(acc.WindowSamples(), 4);
  EXPECT_DOUBLE_EQ(acc.Sum(), 10.0);
  EXPECT_DOUBLE_EQ(acc.Mean(), 2.0);
  EXPECT_DOUBLE_EQ(acc.RollingMean(), 2.5);
  EXPECT_DOUBLE_EQ(acc.Max(), 4.0);
  EXPECT_DOUBLE_EQ(acc.Min(), 0.0);
  EXPECT_NEAR(acc.LazyVariance(), 1.25, 1e-9);
}

TEST(TimingTest, TestAccumulatorInt) {
  timing::Accumulator<int, double, 4> acc;

  for (int i = 0; i < 5; ++i) {
    acc.Add(i);
  }

  EXPECT_EQ(acc.TotalSamples(), 5);
  EXPECT_EQ(acc.WindowSamples(), 4);
  EXPECT_EQ(acc.Sum(), 10);
  EXPECT_DOUBLE_EQ(acc.Mean(), 2.0);
  EXPECT_DOUBLE_EQ(acc.RollingMean(), 2.5);
  EXPECT_EQ(acc.Max(), 4);
  EXPECT_EQ(acc.Min(), 0);
  EXPECT_NEAR(acc.LazyVariance(), 1.25, 1e-9);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
