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
#include <gtest/gtest.h>
#include <memory>

#include <random>
#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/map/common_names.h"

using namespace nvblox;
DEFINE_bool(use_random_seed, false,
            "Whether to use random seed for the stress test");

// We need a non-null pointer when inserting into the hashmap.
TsdfBlock* kDummyPtr = reinterpret_cast<TsdfBlock*>(0x08000000);

std::vector<thrust::pair<Index3D, TsdfBlock*>> createBlocks(
    const int num_blocks, const int start_index = 0) {
  std::vector<thrust::pair<Index3D, TsdfBlock*>> blocks;
  for (int i = 0; i < num_blocks; ++i) {
    blocks.emplace_back(
        thrust::make_pair(Index3D{start_index + i, 0, 0}, kDummyPtr));
  }
  return blocks;
}

// Should be called with a single thread
__global__ void checkHashesEqual(Index3DDeviceHashMapType<TsdfBlock> hash1,
                                 Index3DDeviceHashMapType<TsdfBlock> hash2,
                                 bool* are_equal) {
  *are_equal = true;

  if (hash1.size() != hash2.size()) {
    printf("Not equal due to size mismatch\n");
    *are_equal = false;
    return;
  }

  for (auto pair : hash1) {
    if (hash1.contains(pair.first) && !hash2.contains(pair.first)) {
      printf("Index %i not found in hash2\n", pair.first[0]);
      *are_equal = false;
    }
  }
  for (auto pair : hash2) {
    if (hash2.contains(pair.first) && !hash1.contains(pair.first)) {
      printf("Index %i not found in hash1\n", pair.first[0]);
      *are_equal = false;
    }
  }
}

void testInsertFrom(const int num_blocks, const int capacity) {
  GPULayerView<TsdfBlock> gpu_layer(1);
  gpu_layer.insertBlocksAsync(createBlocks(num_blocks), CudaStreamOwning());
  gpu_layer.flushCache(CudaStreamOwning());
  const GPUHashImpl<TsdfBlock>& src_hash = gpu_layer.getHash();

  GPUHashImpl<TsdfBlock> dst_hash(capacity, CudaStreamOwning());
  dst_hash.initializeFromAsync(src_hash, CudaStreamOwning());

  ASSERT_EQ(src_hash.impl_.size(), num_blocks);
  ASSERT_EQ(dst_hash.impl_.size(), num_blocks);

  device_vector<bool> are_equal_device(1);
  checkHashesEqual<<<1, 1>>>(src_hash.impl_, dst_hash.impl_,
                             are_equal_device.data());

  host_vector<bool> are_equal_host(1);
  are_equal_host.copyFromAsync(are_equal_device, CudaStreamOwning());

  ASSERT_TRUE(are_equal_host[0]);
}

// Test some corner cases
TEST(GpuHashInterface, insertFrom_SameCapacity) { testInsertFrom(10, 10); }
TEST(GpuHashInterface, insertFrom_DifferentCapacity) { testInsertFrom(10, 20); }
TEST(GpuHashInterface, insertFrom_DifferentLargeCapacity) {
  testInsertFrom(1000, 10000);
}
TEST(GpuHashInterface, insertFrom_LargeCapacity) { testInsertFrom(10, 5000); }

// Test different combinations of size/capacity
TEST(GpuHashInterface, insertFrom_combinations) {
  std::vector<int> num_blocks = {0, 1, 10, 100, 100, 1000};
  std::vector<int> capacity = {1, 10, 100, 1000, 10000};

  for (auto n : num_blocks) {
    for (auto c : capacity) {
      if (c >= n) {
        std::cout << " testing n: " << n << " c: " << c << std::endl;
        testInsertFrom(n, c);
      }
    }
  }
}

TEST(GpuHashInterface, initializeFrom_repeatedIndices) {
  constexpr int kCapacity = 100;
  GPULayerView<TsdfBlock> gpu_layer(kCapacity);

  const Index3D index(1, 1, 1);
  gpu_layer.insertBlocksAsync({{index, kDummyPtr}}, CudaStreamOwning());
  gpu_layer.removeBlocksAsync({index}, CudaStreamOwning());
  gpu_layer.insertBlocksAsync({{index, kDummyPtr}}, CudaStreamOwning());
  gpu_layer.flushCache(CudaStreamOwning());

  GPUHashImpl<TsdfBlock> dst_hash(kCapacity, CudaStreamOwning());
  dst_hash.initializeFromAsync(gpu_layer.getHash(), CudaStreamOwning());

  device_vector<bool> are_equal_device(1);
  checkHashesEqual<<<1, 1>>>(gpu_layer.getHash().impl_, dst_hash.impl_,
                             are_equal_device.data());

  host_vector<bool> are_equal_host(1);
  are_equal_host.copyFromAsync(are_equal_device, CudaStreamOwning());

  ASSERT_TRUE(are_equal_host[0]);
}

/// Generate random block indices and insert them into a set. Prune out indices
/// that already exists in the set
std::vector<thrust::pair<Index3D, TsdfBlock*>> generateRandomBlockIndices(
    std::uniform_int_distribution<int>& num_indices_dist,
    std::uniform_int_distribution<int>& index_dist, std::mt19937& rng,
    Index3DSet& inserted) {
  std::vector<Index3D> indices(num_indices_dist(rng));
  std::generate(
      indices.begin(), indices.end(), [&index_dist, &rng]() -> Index3D {
        Index3D idx{index_dist(rng), index_dist(rng), index_dist(rng)};
        return idx;
      });

  std::vector<thrust::pair<Index3D, TsdfBlock*>> block_indices;
  for (auto& index : indices) {
    if (inserted.count(index) == 0) {
      block_indices.push_back({index, kDummyPtr});
      inserted.insert(index);
    }
  }
  return block_indices;
}

TEST(GpuHashInterface, stressTest) {
  // TODO(dtingdahl) make seed and num iterations configurable
  const int num_operations = 1'000'000;

  constexpr int kPrintInterval = 50'000;
  int num_inserted = 0;
  int num_removed = 0;
  int num_flushed = 0;

  GPULayerView<TsdfBlock> gpu_layer(0);
  CudaStreamOwning cuda_stream;

  enum class Operation { kInsert, kRemove, kFlush, kNumOperations };

  // Setup random generator
  std::random_device dev;
  std::unique_ptr<std::mt19937> rng;
  if (FLAGS_use_random_seed) {
    auto seed = dev();
    LOG(INFO) << "Using seed: " << seed;
    rng = std::make_unique<std::mt19937>(seed);

  } else {
    LOG(INFO) << "Using default seed";
    rng = std::make_unique<std::mt19937>();
  }

  constexpr int kMaxNumInsert = 100;
  std::uniform_int_distribution<int> num_insert_dist(0, 100);
  std::uniform_int_distribution<int> num_remove_dist(0, kMaxNumInsert / 2);
  std::uniform_int_distribution<int> index_dist(-100, 100);
  std::uniform_int_distribution<int> operation_dist(
      0, static_cast<int>(static_cast<int>(Operation::kNumOperations) - 1));

  // Keep track on indices we inserted to avoid duplication
  Index3DSet inserted;

  for (int i = 0; i < num_operations; ++i) {
    if ((i % kPrintInterval == 0) || (i == num_operations - 1)) {
      LOG(INFO) << "\nIteration: " << i
                << "\n  GPU layer size: " << gpu_layer.size()
                << "\n Num inserted: " << num_inserted
                << "\n Num removed: " << num_removed
                << "\n Num flushed: " << num_flushed;
    }

    ASSERT_EQ(inserted.size(), gpu_layer.size());
    // Determine which operation we will do
    const Operation operation = static_cast<Operation>(operation_dist(*rng));

    switch (operation) {
      case Operation::kInsert: {
        // Generate a vector of unique indices to insert
        const auto indices_to_insert = generateRandomBlockIndices(
            num_insert_dist, index_dist, *rng, inserted);
        gpu_layer.insertBlocksAsync(indices_to_insert, cuda_stream);
        num_inserted += indices_to_insert.size();
        break;
      }
      case Operation::kRemove: {
        if (inserted.size() > 0) {
          const size_t num_remove =
              std::min<size_t>(inserted.size(), num_remove_dist(*rng));
          std::vector<Index3D> indices_to_remove;
          indices_to_remove.reserve(num_remove);

          // Remove the N first indices
          for (auto itr = inserted.begin();
               itr != inserted.end() && indices_to_remove.size() < num_remove;
               ++itr) {
            indices_to_remove.push_back(*itr);
          }
          for (const auto& index : indices_to_remove) {
            inserted.erase(index);
          }
          gpu_layer.removeBlocksAsync(indices_to_remove, cuda_stream);
          num_removed += indices_to_remove.size();
        }
        break;
      }
      case Operation::kFlush: {
        gpu_layer.flushCache(cuda_stream);
        ++num_flushed;
        break;
      }
      default: {
        LOG(ERROR) << "Invalid operation: " << static_cast<int>(operation);
        ASSERT_TRUE(false);
        break;
      }
    }
  }
}

// Test that we can handle loads of collisions without failing.
// This test requires that stdgpu is patched such that excess_count is set to
// bucket_count, rather than estimated from collision probability.
TEST(GpuHashInterface, collisions) {
  GPULayerView<TsdfBlock> gpu_layer(0);
  CudaStreamOwning cuda_stream;

  constexpr int kNumIndices = 1000;

  Index3DHash hasher;
  size_t prev_hash = 0;
  for (int i = 0; i < kNumIndices; ++i) {
    // Create indices that all hash to the same number
    Index3D index{i * static_cast<int>(Index3DHash::sl), -i, 10};

    // Check that it's really a collision
    size_t hash = hasher(index);
    if (prev_hash != 0) {
      ASSERT_EQ(prev_hash, hash);
    }
    prev_hash = hash;

    // Insert and flush. Assertions within will fail is there's not enough
    // capacity to handle collisions
    gpu_layer.insertBlockAsync({index, kDummyPtr}, cuda_stream);
    gpu_layer.flushCache(cuda_stream);
  }
}

// Special case that used to trigger a CUDA assert
TEST(GpuHashInterface, initializeFrom_SpecialCollisionCase) {
  GPULayerView<TsdfBlock> gpu_layer(10);

  // Create two conflicting indices
  Index3D index_a{0, 0, 10};
  Index3D index_b{static_cast<int>(Index3DHash::sl), -1, 10};

  // Insert both of them
  // A will be inserted at position hash(A) in the internal _values array
  // Since B conflicts, it will be inserted at position hash(A) + offset
  gpu_layer.insertBlockAsync(
      thrust::pair<Index3D, TsdfBlock*>(index_a, kDummyPtr),
      CudaStreamOwning());
  gpu_layer.insertBlockAsync(
      thrust::pair<Index3D, TsdfBlock*>(index_b, kDummyPtr),
      CudaStreamOwning());

  // Remove both of them. They will both remain in the internal _values array,
  // but their _occupied flags will be zeroed.
  gpu_layer.removeBlockAsync(index_b, CudaStreamOwning());
  gpu_layer.removeBlockAsync(index_a, CudaStreamOwning());

  // Re-add B. it will be inserted at position hash(A) (= hash(B)). Hence, there
  // will be two copies of B in _values, one at hash(A) and one at hash(A) +
  // offset
  gpu_layer.insertBlockAsync(
      thrust::pair<Index3D, TsdfBlock*>(index_b, kDummyPtr),
      CudaStreamOwning());
  gpu_layer.flushCache(CudaStreamOwning());

  // Create a copy of the hash. Make sure that we can handle copying of this
  // hash map that contains duplicated entries.
  GPUHashImpl<TsdfBlock> dst_hash(10, CudaStreamOwning());
  dst_hash.initializeFromAsync(gpu_layer.getHash(), CudaStreamOwning());

  EXPECT_EQ(gpu_layer.getHash().impl_.size(), 1);
  EXPECT_EQ(dst_hash.impl_.size(), 1);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
