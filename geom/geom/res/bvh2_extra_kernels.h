#ifndef ROOT_GEOM_BVH2_EXTRA

namespace bvh::v2::extra {

// reusable geometry kernels used in multiple places
// for interaction with BVH2 structures

// determines if a point is inside the bounding box
template <typename T>
bool contains(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   auto min = box.min;
   auto max = box.max;
   return (p[0] >= min[0] && p[0] <= max[0]) && (p[1] >= min[1] && p[1] <= max[1]) &&
          (p[2] >= min[2] && p[2] <= max[2]);
}

// determines the largest squared distance of point to any of the bounding box corners
template <typename T>
auto RmaxSqToNode(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   // construct the 8 corners to get the maximal distance
   const auto minCorner = box.min;
   const auto maxCorner = box.max;
   using Vec3 = bvh::v2::Vec<T, 3>;
   // these are the corners of the bounding box
   const std::array<bvh::v2::Vec<T, 3>, 8> corners{
      Vec3{minCorner[0], minCorner[1], minCorner[2]}, Vec3{minCorner[0], minCorner[1], maxCorner[2]},
      Vec3{minCorner[0], maxCorner[1], minCorner[2]}, Vec3{minCorner[0], maxCorner[1], maxCorner[2]},
      Vec3{maxCorner[0], minCorner[1], minCorner[2]}, Vec3{maxCorner[0], minCorner[1], maxCorner[2]},
      Vec3{maxCorner[0], maxCorner[1], minCorner[2]}, Vec3{maxCorner[0], maxCorner[1], maxCorner[2]}};

   T Rmax_sq{0};
   for (const auto &corner : corners) {
      float R_sq = 0.;
      const auto dx = corner[0] - p[0];
      R_sq += dx * dx;
      const auto dy = corner[1] - p[1];
      R_sq += dy * dy;
      const auto dz = corner[2] - p[2];
      R_sq += dz * dz;
      Rmax_sq = std::max(Rmax_sq, R_sq);
   }
   return Rmax_sq;
};

// determines the minimum squared distance of point to a bounding box ("safey square")
template <typename T>
auto SafetySqToNode(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   T sqDist{0.0};
   for (int i = 0; i < 3; i++) {
      T v = p[i];
      if (v < box.min[i]) {
         sqDist += (box.min[i] - v) * (box.min[i] - v);
      } else if (v > box.max[i]) {
         sqDist += (v - box.max[i]) * (v - box.max[i]);
      }
   }
   return sqDist;
};

} // namespace bvh::v2::extra

#endif