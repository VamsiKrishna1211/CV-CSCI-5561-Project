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

#include "nvblox/geometry/bounding_shape.h"

namespace nvblox {

BoundingShape::BoundingShape() : type_(ShapeType::kSphere) {}

BoundingShape::BoundingShape(BoundingSphere sphere)
    : type_(ShapeType::kSphere) {
  union_.sphere_ = sphere;
}

BoundingShape::BoundingShape(AxisAlignedBoundingBox aabb)
    : type_(ShapeType::kAABB) {
  union_.aabb_ = aabb;
}

BoundingShape::BoundingShape(const nvblox::BoundingShape& value) {
  type_ = value.type_;
  switch (type_) {
    case ShapeType::kSphere:
      union_.sphere_ = value.union_.sphere_;
      break;
    case ShapeType::kAABB:
      union_.aabb_ = value.union_.aabb_;
      break;
    default:
      LOG(FATAL) << "ShapeType not implemented: " << type_;
      break;
  }
}

bool BoundingShape::touchesBlock(const Index3D& block_index,
                                 const float block_size) const {
  switch (type_) {
    case ShapeType::kSphere: {
      return isBlockWithinRadius(block_index, block_size,
                                 union_.sphere_.center(),
                                 union_.sphere_.radius());
    }
    case ShapeType::kAABB: {
      return isBlockTouchedByBoundingBox(block_index, block_size, union_.aabb_);
    }
    default: {
      LOG(FATAL) << "ShapeType not implemented: " << type_;
      break;
    }
  }
}

}  // namespace nvblox
