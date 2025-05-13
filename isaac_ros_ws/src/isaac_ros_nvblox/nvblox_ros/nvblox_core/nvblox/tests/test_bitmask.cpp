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
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <bitset>
#include "nvblox/core/bitmask.h"

namespace nvblox {

enum class TestEnum : int {
  kBit1 = 0,
  kBit2 = 1,
  kBit3 = 2,
  kBit4 = 3,
  kBit63 = 63
};
using TestBitMask = BitMask<TestEnum>;

TEST(BitMask, Assignment) {
  const TestBitMask mask1(TestEnum::kBit2);
  const TestBitMask mask2 = mask1;
  EXPECT_EQ(mask1, TestEnum::kBit2);
  EXPECT_EQ(mask2, TestEnum::kBit2);
}

TEST(BitMask, Construct) {
  const TestBitMask mask2 = TestEnum::kBit4;
  EXPECT_EQ(mask2.get(), 0b1000);
}

TEST(BitMask, LogicalOr) {
  const TestBitMask mask1 = TestEnum::kBit1;
  const TestBitMask mask2 = TestEnum::kBit4;
  const TestBitMask mask3 = TestEnum::kBit63;
  const TestBitMask mask_or = mask1 | mask2 | mask3;

  // Check that the first and forth bits are set
  EXPECT_EQ(mask_or.get() & 0b0001, 1);
  EXPECT_EQ(mask_or.get() & 0b0010, 0);
  EXPECT_EQ(mask_or.get() & 0b0100, 0);
  EXPECT_EQ(mask_or.get() & 0b1000, 8);
  EXPECT_EQ(mask_or.get() & (1LL << 63LL), (1LL << 63LL));

  // ASSIGN-or
  TestBitMask mask_assign_or = mask1;
  mask_assign_or |= mask2 | mask3;

  EXPECT_EQ(mask_assign_or, mask_or);
}

TEST(BitMask, BooleanAnd) {
  TestBitMask mask = TestEnum::kBit2;
  EXPECT_EQ(mask & TestEnum::kBit1, 0);
  EXPECT_GT(mask & TestEnum::kBit2, 0);
  EXPECT_EQ(mask & TestEnum::kBit3, 0);
}

}  // namespace nvblox
int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
