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
#include <nvblox/mesh/mesh.h>

namespace nvblox {

// template <typename T>
// void expandByFactorIfTooSmallAsync(const size_t required_size,
//                                    const float expansion_factor,
//                                    unified_vector<T>* vec,
//                                    const CudaStream& stream) {
//   if (vec->size() < required_size) {
//     const size_t new_size = static_cast<size_t>(
//         expansion_factor * static_cast<float>(required_size));
//     vec->reserveAsync(new_size, stream);
//   }
// }

Mesh Mesh::fromLayer(const BlockLayer<MeshBlock>& layer,
                     const CudaStream& cuda_stream) {
  Mesh mesh;

  // Keep track of the vertex index.
  int next_index = 0;

  // Iterate over every block in the layer.
  const std::vector<Index3D> indices = layer.getAllBlockIndices();

  // Allocate Staging areas up-front
  unified_vector<Vector3f> vertices(MemoryType::kHost);
  unified_vector<Vector3f> normals(MemoryType::kHost);
  unified_vector<Color> colors(MemoryType::kHost);
  unified_vector<int> triangles(MemoryType::kHost);

  // Loop over mesh blocks bringing them back from the GPU 1 by 1.
  for (const Index3D& index : indices) {
    MeshBlock::ConstPtr block = layer.getBlockAtIndex(index);

    // Reduce the frequency of allocation by expanding by some factor (1.5)
    expandBuffersIfRequired(block->vertices.size(), cuda_stream, &vertices);
    expandBuffersIfRequired(block->normals.size(), cuda_stream, &normals);
    expandBuffersIfRequired(block->colors.size(), cuda_stream, &colors);
    expandBuffersIfRequired(block->triangles.size(), cuda_stream, &triangles);

    // First copy everything from GPU
    vertices.copyFromAsync(block->vertices, cuda_stream);
    normals.copyFromAsync(block->normals, cuda_stream);
    colors.copyFromAsync(block->colors, cuda_stream);
    triangles.copyFromAsync(block->triangles, cuda_stream);
    cuda_stream.synchronize();

    // Append to the mesh elements
    mesh.vertices.resize(mesh.vertices.size() + vertices.size());
    std::copy(vertices.begin(), vertices.end(),
              mesh.vertices.begin() + next_index);

    mesh.normals.resize(mesh.normals.size() + normals.size());
    std::copy(normals.begin(), normals.end(),
              mesh.normals.begin() + next_index);

    mesh.colors.resize(mesh.colors.size() + colors.size());
    std::copy(colors.begin(), colors.end(), mesh.colors.begin() + next_index);

    // Our simple mesh implementation has:
    // - per vertex colors
    // - per vertex normals
    CHECK((vertices.size() == normals.size()) || (normals.size() == 0));
    CHECK((vertices.size() == vertices.size()) || (colors.size() == 0));

    std::vector<int> triangle_indices(triangles.size());
    // Increment all triangle indices.
    std::transform(triangles.begin(), triangles.end(), triangle_indices.begin(),
                   std::bind2nd(std::plus<int>(), next_index));

    mesh.triangles.insert(mesh.triangles.end(), triangle_indices.begin(),
                          triangle_indices.end());

    next_index += vertices.size();
  }

  return mesh;
}

}  // namespace nvblox
