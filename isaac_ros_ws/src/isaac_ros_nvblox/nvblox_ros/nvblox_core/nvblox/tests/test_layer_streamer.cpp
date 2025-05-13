#include <numeric>

#include <gtest/gtest.h>

#include "nvblox/core/hash.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/serialization/layer_streamer.h"

#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

template <class LayerType>
class LayerStreamerTestFixture : public ::testing::Test {};
using LayerStreamerTypes = ::testing::Types<MeshLayerStreamerOldestBlocks,
                                            TsdfLayerStreamerOldestBlocks>;

TYPED_TEST_SUITE(LayerStreamerTestFixture, LayerStreamerTypes);

template <class LayerType>
class SimpleLayerStreamer : public LayerStreamerBase<LayerType> {
 public:
  SimpleLayerStreamer() = default;
  virtual ~SimpleLayerStreamer() = default;

  // To be able to test the internal structure
  const Index3DSet& index_set() const {
    return LayerStreamerBase<LayerType>::index_set_;
  }

 protected:
  // For our test streamer the priority is equal to the x-dimension of the
  // index.
  std::vector<float> computePriorities(
      const std::vector<Index3D>& block_indices) const override {
    std::vector<float> priorities;
    std::transform(block_indices.begin(), block_indices.end(),
                   std::back_inserter(priorities), [&](const Index3D& block) {
                     return computePriority(block);
                   });
    return priorities;
  }
  float computePriority(const Index3D& block) const {
    return static_cast<float>(block.x());
  }
};

template <class LayerType>
class SimpleLayerStreamerTestFixture : public ::testing::Test {};
using SimpleLayerStreamerTypes =
    ::testing::Types<SimpleLayerStreamer<MeshLayer>,
                     SimpleLayerStreamer<TsdfLayer>>;
TYPED_TEST_SUITE(SimpleLayerStreamerTestFixture, SimpleLayerStreamerTypes);

constexpr int kDefaultMinRandomIndex = -1000;
constexpr int kDefaultMaxRandomIndex = 1000;
template <class LayerStreamerType>
void fillWithRandomIndices(const int num_indices,
                           LayerStreamerType* layer_streamer_ptr,
                           const int min_index = kDefaultMinRandomIndex,
                           const int max_index = kDefaultMaxRandomIndex) {
  std::vector<Index3D> indices(num_indices);
  std::generate(indices.begin(), indices.end(), [&]() {
    return test_utils::getRandomIndex3dInRange(min_index, max_index);
  });
  layer_streamer_ptr->markIndicesCandidates(indices);
}

TYPED_TEST(SimpleLayerStreamerTestFixture, SimplePriorityTest) {
  // Add a bunch of blocks with random indices
  constexpr int kNumInStreamer = 100;
  constexpr int kNumRequested = 95;

  // Make a mesh streamer to test and fill it with stuff
  TypeParam layer_streamer;
  fillWithRandomIndices(kNumInStreamer, &layer_streamer);

  // Get the high priority block indices
  const std::vector<Index3D> ordered_indices =
      layer_streamer.getNBlocks(kNumRequested);
  EXPECT_EQ(ordered_indices.size(), kNumRequested);

  // Go through the indices and check that the x-elements are decreasing
  int last_x_value = std::numeric_limits<int>::max();
  for (size_t i = 0; i < ordered_indices.size(); i++) {
    EXPECT_LE(ordered_indices[i].x(), last_x_value);
    last_x_value = ordered_indices[i].x();
  }

  // Check that the remaining indices have x-elements less the the smallest in
  // the returned set
  for (const Index3D& idx : layer_streamer.index_set()) {
    EXPECT_LT(idx.x(), last_x_value);
  }

  // Check that the set is now empty
  EXPECT_EQ(layer_streamer.index_set().size(), kNumInStreamer - kNumRequested);
}

TYPED_TEST(SimpleLayerStreamerTestFixture, RequestMoreThanAvailable) {
  constexpr int kNumInStreamer = 50;
  constexpr int kNumRequested = 100;
  // Fill
  TypeParam layer_streamer;
  fillWithRandomIndices<TypeParam>(kNumInStreamer, &layer_streamer);
  // Request more
  const std::vector<Index3D> block_indices =
      layer_streamer.getNBlocks(kNumRequested);

  EXPECT_EQ(block_indices.size(), kNumInStreamer);
  EXPECT_EQ(layer_streamer.index_set().size(), 0);
}

TYPED_TEST(SimpleLayerStreamerTestFixture, RequestZero) {
  constexpr int kNumInStreamer = 50;
  // Fill
  TypeParam layer_streamer;
  fillWithRandomIndices<TypeParam>(kNumInStreamer, &layer_streamer);
  // Request 0
  const std::vector<Index3D> block_indices = layer_streamer.getNBlocks(0);

  EXPECT_TRUE(block_indices.empty());
  EXPECT_EQ(layer_streamer.index_set().size(), kNumInStreamer);
}

std::vector<Index3D> getUniqueIndex3DVectors(
    const int num_block_indices, const int min_index = kDefaultMinRandomIndex,
    const int max_index = kDefaultMaxRandomIndex) {
  // Loops trying to fill a set with as many indices as requested
  Index3DSet index_set;
  constexpr int kMaxNumberOfTries = 1e7;
  int num_tries = 0;
  while (static_cast<int>(index_set.size()) < num_block_indices) {
    index_set.insert(test_utils::getRandomIndex3dInRange(min_index, max_index));
    CHECK_LT(num_tries++, kMaxNumberOfTries);
  }
  return std::vector<Index3D>(index_set.begin(), index_set.end());
}
template <class StreamerType>
void fillWithRandomUniqueIndices(const int num_indices,
                                 StreamerType* layer_streamer_ptr,
                                 const int min_index = kDefaultMinRandomIndex,
                                 const int max_index = kDefaultMaxRandomIndex) {
  layer_streamer_ptr->markIndicesCandidates(
      getUniqueIndex3DVectors(num_indices, min_index, max_index));
}

template <typename StreamerType>
class LayerStreamerOldestBlocksTest : public StreamerType {
 public:
  LayerStreamerOldestBlocksTest() = default;
  virtual ~LayerStreamerOldestBlocksTest() = default;

  // To be able to test the internal structure
  const Index3DSet& index_set() const { return StreamerType::index_set_; }

  const BlockIndexToLastPublishedIndexMap& last_published_map() const {
    return StreamerType::last_published_map_;
  }
};

template <class LayerType>
class LayerStreamerOldestBlocksTestFixture : public ::testing::Test {};
using LayerStreamerOldestBlocksTypes = ::testing::Types<
    LayerStreamerOldestBlocksTest<MeshLayerStreamerOldestBlocks>,
    LayerStreamerOldestBlocksTest<TsdfLayerStreamerOldestBlocks>>;
TYPED_TEST_SUITE(LayerStreamerOldestBlocksTestFixture,
                 LayerStreamerOldestBlocksTypes);

int getSizeOfIntersection(const std::vector<Index3D>& indices_1,
                          const std::vector<Index3D>& indices_2) {
  std::vector<Index3D> indices_1_sorted = indices_1;
  std::vector<Index3D> indices_2_sorted = indices_2;
  VectorCompare<Index3D> comp;
  std::sort(indices_1_sorted.begin(), indices_1_sorted.end(), comp);
  std::sort(indices_2_sorted.begin(), indices_2_sorted.end(), comp);
  std::vector<Index3D> intersection;
  std::set_intersection(indices_1_sorted.begin(), indices_1_sorted.end(),
                        indices_2_sorted.begin(), indices_2_sorted.end(),
                        std::back_inserter(intersection), comp);
  return intersection.size();
}

TYPED_TEST(LayerStreamerOldestBlocksTestFixture, LayerStreamerOldestBlocks) {
  // Create and fill with random indices
  constexpr int kNumInStreamer = 100;
  TypeParam layer_streamer;
  fillWithRandomUniqueIndices<TypeParam>(kNumInStreamer, &layer_streamer);
  EXPECT_EQ(layer_streamer.numCandidates(), kNumInStreamer);

  // Get the first half of the indices (all indices have the same priority at
  // the moment so this should just be random).
  const std::vector<Index3D> first_half =
      layer_streamer.getNBlocks(50, BlockExclusionParams());
  EXPECT_EQ(first_half.size(), 50);
  // Get the second half
  const std::vector<Index3D> second_half =
      layer_streamer.getNBlocks(50, BlockExclusionParams());
  EXPECT_EQ(second_half.size(), 50);
  EXPECT_EQ(layer_streamer.numCandidates(), 0);

  // Check that the halves are distict
  EXPECT_EQ(getSizeOfIntersection(first_half, second_half), 0);

  // Add them back in
  layer_streamer.markIndicesCandidates(first_half);
  layer_streamer.markIndicesCandidates(second_half);

  // Get the two halves again
  const std::vector<Index3D> first_half_2 =
      layer_streamer.getNBlocks(50, BlockExclusionParams());
  const std::vector<Index3D> second_half_2 =
      layer_streamer.getNBlocks(50, BlockExclusionParams());

  // Check that come out in the same order.
  EXPECT_EQ(getSizeOfIntersection(first_half, first_half_2), 50);
  EXPECT_EQ(getSizeOfIntersection(second_half, second_half_2), 50);
  EXPECT_EQ(getSizeOfIntersection(first_half, second_half_2), 0);
  EXPECT_EQ(getSizeOfIntersection(first_half_2, second_half_2), 0);
}

TEST(TsdfLayerStreamerOldestBlocks, SerializeNBytes) {
  // Create a TSDF layer
  primitives::Scene scene = test_utils::getSphereInBox();
  constexpr float kVoxelSizeM = 0.05;
  constexpr int kMaxDistVox = 4;
  constexpr float kMaxDistM = static_cast<float>(kMaxDistVox) * kVoxelSizeM;
  TsdfLayer tsdf_layer(kVoxelSizeM, MemoryType::kUnified);
  scene.generateLayerFromScene(kMaxDistM, &tsdf_layer);

  // Get size in bytes of layer
  const size_t layer_size = tsdf_layer.numAllocatedBlocks() * sizeof(TsdfBlock);
  const size_t bytes_to_serialize = layer_size / 2;
  ASSERT_GE(bytes_to_serialize, 0);

  // Serializer the layer
  TsdfLayerStreamerOldestBlocks layer_streamer;
  auto serialized_layer = layer_streamer.getNBytesOfSerializedBlocks(
      bytes_to_serialize, tsdf_layer, BlockExclusionParams(),
      CudaStreamOwning());
  const size_t serialized_size =
      serialized_layer->voxels.size() * sizeof(TsdfVoxel);

  EXPECT_LE(serialized_size, bytes_to_serialize);
}

TEST(MeshLayerStreamerOldestBlocks, StreamNBytes) {
  // Create a test scene and mesh it
  primitives::Scene scene = test_utils::getSphereInBox();
  constexpr float kVoxelSizeM = 0.05;
  constexpr int kMaxDistVox = 4;
  constexpr float kMaxDistM = static_cast<float>(kMaxDistVox) * kVoxelSizeM;
  TsdfLayer tsdf_layer(kVoxelSizeM, MemoryType::kUnified);
  scene.generateLayerFromScene(kMaxDistM, &tsdf_layer);
  MeshIntegrator mesh_integrator;
  MeshLayer mesh_layer(tsdf_layer.block_size(), MemoryType::kDevice);
  mesh_integrator.integrateMeshFromDistanceField(tsdf_layer, &mesh_layer);

  // Create streamer
  MeshLayerStreamerOldestBlocks layer_streamer;

  // Get the total size of the mesh
  const std::vector<MeshBlock*> blocks = mesh_layer.getAllBlockPointers();
  const int total_bytes =
      std::accumulate(blocks.begin(), blocks.end(), 0,
                      [](const int sum, const MeshBlock* block) {
                        return sum + sizeInBytes(block);
                      });
  LOG(INFO) << "mesh has : " << total_bytes << " bytes";

  // Fill with all indices
  layer_streamer.markIndicesCandidates(mesh_layer.getAllBlockIndices());

  // Get half of the indices
  const int half_mesh_bytes = total_bytes / 2;
  const std::vector<Index3D> half_blocks = layer_streamer.getNBytesOfBlocks(
      half_mesh_bytes, mesh_layer, BlockExclusionParams());

  // Check that the total size of returned mesh blocks is the right size.
  const int returned_bytes = std::accumulate(
      half_blocks.begin(), half_blocks.end(), 0,
      [&mesh_layer](const int sum, const Index3D block_idx) {
        return sum + sizeInBytes(mesh_layer.getBlockAtIndex(block_idx).get());
      });
  LOG(INFO) << "mesh streamer returned: " << returned_bytes << " bytes";
  CHECK_LT(returned_bytes, (total_bytes + 1) / 2);
  const float proportion_returned =
      static_cast<float>(returned_bytes) / static_cast<float>(total_bytes);
  constexpr float kAllowableError = 0.05;  // 5% off 50% is fine
  CHECK_NEAR(proportion_returned, 0.5, kAllowableError);

  // Lets get out the remaining parts of the mesh
  const std::vector<Index3D> second_half_blocks =
      layer_streamer.getNBytesOfBlocks(total_bytes, mesh_layer,
                                       BlockExclusionParams());
  const int second_returned_bytes =
      std::accumulate(second_half_blocks.begin(), second_half_blocks.end(), 0,
                      [&mesh_layer](const int sum, const Index3D block_idx) {
                        auto block = mesh_layer.getBlockAtIndex(block_idx);
                        return sum + sizeInBytes(block.get());
                      });
  LOG(INFO) << "mesh streamer returned: " << second_returned_bytes
            << " more bytes";

  std::vector<Index3D> all_returned_blocks = half_blocks;
  all_returned_blocks.insert(all_returned_blocks.end(),
                             second_half_blocks.begin(),
                             second_half_blocks.end());

  const int num_blocks_returned = getSizeOfIntersection(
      all_returned_blocks, mesh_layer.getAllBlockIndices());
  const int total_num_blocks_in_layer = mesh_layer.numAllocatedBlocks();
  EXPECT_EQ(num_blocks_returned, total_num_blocks_in_layer);
}

TYPED_TEST(LayerStreamerOldestBlocksTestFixture, DeallocatedBlock) {
  // Create layer
  constexpr float kBlockSizeM = 0.4f;
  typename TypeParam::LayerType layer(kBlockSizeM, MemoryType::kDevice);

  // Get a random list of (non-existing) block indices.
  constexpr int kNumIndices = 100;
  const std::vector<Index3D> indices = getUniqueIndex3DVectors(kNumIndices);

  // Layer streamer, and add indices
  TypeParam layer_streamer;
  layer_streamer.markIndicesCandidates(indices);
  EXPECT_EQ(layer_streamer.index_set().size(), kNumIndices);

  // Request bytes (which will fail to lookup blocks)
  constexpr int kNumBytesRequested = 1e6;
  const auto returned_indices = layer_streamer.getNBytesOfBlocks(
      kNumBytesRequested, layer, BlockExclusionParams());

  // Check that 0 blocks are returned
  EXPECT_EQ(returned_indices.size(), 0);
  // Check that the invalid blocks got deleted
  EXPECT_EQ(layer_streamer.index_set().size(), 0);
}

template <class LayerType>
class ExcludeAllBlocksLayerStreamer : public SimpleLayerStreamer<LayerType> {
 public:
  ExcludeAllBlocksLayerStreamer() : SimpleLayerStreamer<LayerType>() {
    // Add a functor that excludes all blocks
    std::vector<ExcludeBlockFunctor> exclude_block_functors;
    exclude_block_functors.push_back(
        [](const Index3D&) -> bool { return true; });
    LayerStreamerBase<LayerType>::setExclusionFunctors(exclude_block_functors);
  }
  virtual ~ExcludeAllBlocksLayerStreamer() = default;

  // For checking the exclusion functions
  std::vector<ExcludeBlockFunctor> exclude_block_functors() const {
    return LayerStreamerBase<LayerType>::exclude_block_functors_;
  }

 protected:
};

template <class LayerType>
class ExcludeAllBlocksLayerStreamerTestFixture : public ::testing::Test {};
using ExcludeAllBlocksLayerStreamerTypes =
    ::testing::Types<ExcludeAllBlocksLayerStreamer<MeshLayer>,
                     ExcludeAllBlocksLayerStreamer<TsdfLayer>>;
TYPED_TEST_SUITE(ExcludeAllBlocksLayerStreamerTestFixture,
                 ExcludeAllBlocksLayerStreamerTypes);

TYPED_TEST(ExcludeAllBlocksLayerStreamerTestFixture, ExcludeBlocksSimpleTest) {
  constexpr int kNumInStreamer = 50;
  // Fill
  TypeParam layer_streamer;
  fillWithRandomIndices(kNumInStreamer, &layer_streamer);

  // Check that we successfully added an exclusion function
  EXPECT_EQ(layer_streamer.exclude_block_functors().size(), 1);
  EXPECT_TRUE(layer_streamer.exclude_block_functors()[0](Index3D::Zero()));

  // Get the blocks back
  const std::vector<Index3D> blocks = layer_streamer.getNBlocks(kNumInStreamer);

  // Check that all blocks have been excluded
  EXPECT_EQ(blocks.size(), 0);

  // Check that they've also been removed internally
  EXPECT_EQ(layer_streamer.index_set().size(), 0);
}

std::vector<Index3D> getCubeOfBlockIndices(const int side_length_in_blocks) {
  std::vector<Index3D> block_indices;
  for (int x = 0; x < side_length_in_blocks; x++) {
    for (int y = 0; y < side_length_in_blocks; y++) {
      for (int z = 0; z < side_length_in_blocks; z++) {
        block_indices.push_back(Index3D(x, y, z));
      }
    }
  }
  return block_indices;
}

TYPED_TEST(LayerStreamerOldestBlocksTestFixture, ExcludeBlocksAboveHeight) {
  constexpr float kBlockSizeM = 0.1;
  TypeParam layer_streamer;

  // Turn on above height exclusion
  constexpr float kBlockHeightLimitM = 0.55;

  // Make a cube of blocks 1x1x1m
  constexpr int kCubeSideLengthInBlocks = 10;
  const std::vector<Index3D> block_indices =
      getCubeOfBlockIndices(kCubeSideLengthInBlocks);

  // Add them to the streamer
  layer_streamer.markIndicesCandidates(block_indices);

  // Get back the blocks
  const auto streamable_blocks = layer_streamer.getNBlocks(
      block_indices.size(),
      BlockExclusionParams{.exclusion_height_m = kBlockHeightLimitM,
                           .block_size_m = kBlockSizeM});

  // Check that some were actually excluded
  EXPECT_LT(streamable_blocks.size(), block_indices.size());

  // Go through the returned blocks and check that the returned blocks are all
  // below the limit.
  for (const auto& block_idx : streamable_blocks) {
    const float min_z_value = block_idx.z() * kBlockSizeM;
    EXPECT_LT(min_z_value, kBlockHeightLimitM);
  }
  // Check that the streamer is empty
  EXPECT_EQ(layer_streamer.numCandidates(), 0);
}

TYPED_TEST(LayerStreamerOldestBlocksTestFixture, ExcludeBlocksOutsideRadius) {
  constexpr float kBlockSizeM = 0.1;
  TypeParam layer_streamer;

  // Turn on above height exclusion
  constexpr float kBlockRadiusLimitM = 0.5;

  // Make a cube of blocks 1x1x1m
  constexpr int kCubeSideLengthInBlocks = 10;
  const std::vector<Index3D> block_indices =
      getCubeOfBlockIndices(kCubeSideLengthInBlocks);

  // Add them to the streamer
  layer_streamer.markIndicesCandidates(block_indices);

  // The center around which to perform exclusion
  const Vector3f exclusion_center_m =
      (kCubeSideLengthInBlocks / 2.0) * kBlockSizeM * Vector3f::Ones();

  // Get back the blocks
  const auto streamable_blocks = layer_streamer.getNBlocks(
      block_indices.size(),
      BlockExclusionParams{.exclusion_center_m = exclusion_center_m,
                           .exclusion_radius_m = kBlockRadiusLimitM,
                           .block_size_m = kBlockSizeM});
  EXPECT_LT(streamable_blocks.size(), block_indices.size());

  // Check blocks not outside the exclusion radius.
  std::vector<float> center_distances;
  std::for_each(streamable_blocks.begin(), streamable_blocks.end(),
                [&](const Index3D& idx) {
                  const Vector3f block_center =
                      getCenterPositionFromBlockIndex(kBlockSizeM, idx);
                  const float radius =
                      (block_center - exclusion_center_m).norm();
                  EXPECT_LT(radius, kBlockRadiusLimitM);
                });
  // Check that the streamer is empty
  EXPECT_EQ(layer_streamer.numCandidates(), 0);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
