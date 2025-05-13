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
#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "nvblox/utils/rates.h"

using namespace nvblox;

TEST(RatesTest, KnownRateTest) {
  // This test can sometimes be flaky when the system is under load. We
  // therefore allow for some flakiness.
  constexpr int kMaxNumRetries = 10;
  bool test_passed = false;

  for (int i = 0; i < kMaxNumRetries && !test_passed; ++i) {
    constexpr int kNumTicks = 100;
    constexpr float kRateHz = 100;

    constexpr float sleep_duration_s = 1.0f / static_cast<float>(kRateHz);
    constexpr int kSecondsToNanoSeconds = 1e9;
    constexpr int sleep_duration_ns =
        static_cast<int>(sleep_duration_s * kSecondsToNanoSeconds);

    for (int j = 0; j < kNumTicks; j++) {
      timing::Rates::tick("rates_test/test_1");
      timing::Rates::tick("rates_test/test_2");
      std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_duration_ns));
    }
    LOG(INFO) << timing::Rates::Print();
    const float estimated_rate =
        timing::Rates::getMeanRateHz("rates_test/test_1");
    const float estimated_rate_2 =
        timing::Rates::getMeanRateHz("rates_test/test_2");
    LOG(INFO) << "We tried to loop at: " << kRateHz
              << " Hz, and measured: " << estimated_rate << " Hz";
    // NOTE(alexmillane): sleep_for can be quite innaccurate. Especially if
    // there's high CPU usage on the machine running the test. This is why we
    // have a large tolerance here. We may even have to increase it further.
    constexpr float kRateToleranceHz = 20.0;
    // NOTE(alexmillane): On the jetson I observed differences of ~0.001 between
    // the two timers, during high loads. So I over estimate this parameter to
    // not generate a flakey test.
    constexpr float kRateEps = 0.1;

    const bool rate_within_tolerance =
        std::fabs(estimated_rate - kRateHz) < kRateToleranceHz;
    const bool rates_near =
        std::fabs(estimated_rate - estimated_rate_2) < kRateEps;

    EXPECT_TRUE(rate_within_tolerance);
    EXPECT_TRUE(rates_near);

    test_passed = rate_within_tolerance && rates_near;

    if (!test_passed) {
      LOG(WARNING) << "KnownRateTest is flaky. Iteration: " << i << " out of "
                   << kMaxNumRetries;
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  }
}

TEST(RatesTest, NonExistingTicker) {
  timing::Rates::tick("existing_tag");
  const float existing_rate = timing::Rates::getMeanRateHz("existing_tag");

  constexpr float kEps = 1e-6;
  EXPECT_TRUE(timing::Rates::exists("existing_tag"));
  EXPECT_NEAR(existing_rate, 0.0f, kEps);

  const float non_existing_rate =
      timing::Rates::getMeanRateHz("non_existing_tag");
  EXPECT_FALSE(timing::Rates::exists("none_existing_tag"));
  EXPECT_NEAR(non_existing_rate, 0.0f, kEps);
}

// A class with identifyable default construction to help test the circular
// buffer.
struct DefaultConstructableStruct {
  DefaultConstructableStruct() = default;

  int member_1 = 1;
  int member_2 = 2;
  int member_3 = 3;
};

TEST(RatesTest, CircularBufferEmptyTest) {
  timing::CircularBuffer<DefaultConstructableStruct, 10> circular_buffer;

  EXPECT_TRUE(circular_buffer.empty());

  // Check that the circular buffer returns a default constructed struct if
  // empty
  auto newest = circular_buffer.newest();
  EXPECT_EQ(newest.member_1, 1);
  EXPECT_EQ(newest.member_2, 2);
  EXPECT_EQ(newest.member_3, 3);
}

TEST(RatesTest, CircularBufferTest) {
  timing::CircularBuffer<int, 10> circular_buffer;

  EXPECT_TRUE(circular_buffer.empty());
  EXPECT_FALSE(circular_buffer.full());

  circular_buffer.push(1);
  EXPECT_FALSE(circular_buffer.empty());
  EXPECT_FALSE(circular_buffer.full());
  EXPECT_EQ(circular_buffer.size(), 1);

  EXPECT_EQ(circular_buffer.newest(), 1);
  EXPECT_EQ(circular_buffer.oldest(), 1);

  circular_buffer.push(2);
  EXPECT_FALSE(circular_buffer.empty());
  EXPECT_FALSE(circular_buffer.full());
  EXPECT_EQ(circular_buffer.size(), 2);

  EXPECT_EQ(circular_buffer.newest(), 2);
  EXPECT_EQ(circular_buffer.oldest(), 1);

  // Fill the buffer (but stop just before wrap around)
  for (int i = 3; i <= 10; i++) {
    circular_buffer.push(i);
  }
  EXPECT_TRUE(circular_buffer.full());
  EXPECT_EQ(circular_buffer.size(), 10);

  EXPECT_EQ(circular_buffer.newest(), 10);
  EXPECT_EQ(circular_buffer.oldest(), 1);

  // Wrap around
  circular_buffer.push(11);
  EXPECT_TRUE(circular_buffer.full());

  EXPECT_EQ(circular_buffer.newest(), 11);
  EXPECT_EQ(circular_buffer.oldest(), 2);

  // Fill the buffer with a bunch more values
  constexpr int kNumExtraValues = 37;
  for (int i = 0; i < kNumExtraValues; i++) {
    circular_buffer.push(i);
  }
  EXPECT_TRUE(circular_buffer.full());
  EXPECT_EQ(circular_buffer.size(), 10);

  EXPECT_EQ(circular_buffer.newest(), kNumExtraValues - 1);
  EXPECT_EQ(circular_buffer.newest() - circular_buffer.oldest(), 9);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
