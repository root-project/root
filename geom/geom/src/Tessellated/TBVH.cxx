// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TBVH
\ingroup Geometry_classes

Partitioning structure to improve processing time for TGeoTessellated's
navigation functions. Uses bounding volume hierarchy structure contained in
geom/geom/inc/bvh/v2
*/

#include "Tessellated/TBVH.h"

#include <iostream>

// Substantial copying from TGeoParallelWorld.cxx
namespace Tessellated {
// ClassImp(TBVH)

namespace TBVHInternal {
////////////////////////////////////////////////////////////////////////////////
/// This code is taken from Dr. Sandro Wenzel's commit to include 
/// the BVH to TGeoParallelWorld 

////////////////////////////////////////////////////////////////////////////////
/// Boilerplate code to find the closest triangle to a point
/// structure keeping cost (value) for a BVH index
struct BVHPrioElement {
   size_t bvh_node_id;
   Double_t value;
};

/// A priority queue for BVHPrioElement with an additional clear method
/// for quick reset
template <typename Comparator>
class BVHPrioQueue : public std::priority_queue<BVHPrioElement, std::vector<BVHPrioElement>, Comparator> {
public:
   using std::priority_queue<BVHPrioElement, std::vector<BVHPrioElement>,
                             Comparator>::priority_queue; // constructor inclusion

   // convenience method to quickly clear/reset the queue (instead of having to pop one by one)
   void clear() { this->c.clear(); }
};


// determines if a point is inside the bounding box
template <typename T>
bool Contains(bvh::v2::BBox<T, 3> const &box, bvh::v2::Vec<T, 3> const &p)
{
   auto min = box.min;
   auto max = box.max;
   return (p[0] >= min[0] && p[0] <= max[0]) && (p[1] >= min[1] && p[1] <= max[1]) &&
          (p[2] >= min[2] && p[2] <= max[2]);
}

// determines the mininum squared distance of point to a bounding box ("safey square")
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
}

} // namespace TBVHInternal

////////////////////////////////////////////////////////////////////////////////
/// Reset the class members

void TBVH::ResetInternalState()
{
   fBVH.reset(nullptr);
}

////////////////////////////////////////////////////////////////////////////////
/// Build the BVH

void TBVH::BuildBVH()
{
   ResetInternalState();
   const std::vector<TGeoTriangle> &triangles = fMesh->Triangles();
   const size_t nTriangles{triangles.size()};
   
   // Get triangle centers and bounding boxes (required for BVH builder)
   std::vector<BBox> bboxes;
   bboxes.resize(nTriangles);
   std::vector<Vec3> centers(nTriangles);
   size_t i{0};
   for (const auto &triangle : triangles) {
      const ROOT::Math::XYZVector &a = triangle.Point(0);
      const ROOT::Math::XYZVector &b = triangle.Point(1);
      const ROOT::Math::XYZVector &c = triangle.Point(2);
      bboxes[i].min[0] = std::min({a.X(),b.X(),c.X()});
      bboxes[i].min[1] = std::min({a.Y(),b.Y(),c.Y()});
      bboxes[i].min[2] = std::min({a.Z(),b.Z(),c.Z()});
      bboxes[i].max[0] = std::max({a.X(),b.X(),c.X()});
      bboxes[i].max[1] = std::max({a.Y(),b.Y(),c.Y()});
      bboxes[i].max[2] = std::max({a.Z(),b.Z(),c.Z()});
      centers[i] = bboxes[i].get_center();
      ++i;
   }

   typename bvh::v2::DefaultBuilder<Node>::Config config;
   config.quality = fBVHQuality;
   auto bvh = bvh::v2::DefaultBuilder<Node>::build(bboxes, centers, config);
   auto bvhptr = new Bvh{};
   *bvhptr = std::move(bvh);
   fBVH.reset(bvhptr);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the triangle mesh (triangle mesh is owned by TGeoTessellated) and build the bvh

void TBVH::SetTriangleMesh(const TGeoTriangleMesh *trianglemesh)
{
   TPartitioningI::SetTriangleMesh(trianglemesh);
   BuildBVH();
}

////////////////////////////////////////////////////////////////////////////////
/// Set the BVH Quality and rebuild the bvh
///
/// \param[in] quality Valid options are
///   1. bvh::v2::DefaultBuilder<Node>::Quality::Low
///   2. bvh::v2::DefaultBuilder<Node>::Quality::Medium
///   3. bvh::v2::DefaultBuilder<Node>::Quality::High

void TBVH::SetBVHQuality(bvh::v2::DefaultBuilder<Node>::Quality quality)
{
   fBVHQuality = quality;
   if (GetTriangleMesh() != nullptr) {
      BuildBVH();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Is the point contained by the geometry stored in this octree
///
/// \param[in] point to be tested for inside/outside state
/// \return Bool_t
///

Bool_t TBVH::IsPointContained(const ROOT::Math::XYZVector &point) const
{
   // Shoot test ray in arbitrary direction. Check for intersected triangle. If intersection occured, check
   // projection of testdirection and triangle normal
   ROOT::Math::XYZVector testdir{0, 0, 1};
   auto ray = Ray{
      Vec3(point.X(), point.Y(), point.Z()),      // Ray origin
      Vec3(testdir.X(), testdir.Y(), testdir.Z()) // Ray direction
   };
   const TGeoTriangle *triangle = GetIntersectedTriangle(ray);
   if (triangle != nullptr) {
      return triangle->Normal().Dot(testdir) > 0 || ray.tmax < TGeoShape::Tolerance();
   } else {
      return kFALSE;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Get the closest Triangle object to point. This is also almost taken 1:1 
/// from Dr. Sandro Wenzel's BVH implementation for
/// TGeoParallelWorld::GetBVHSafetyCandidates, besides the computation of 
///
/// \param[in] point to which the closest triangle needs to be found
/// \return TGeoTriangleMesh::ClosestTriangle_t
///

TGeoTriangleMesh::ClosestTriangle_t TBVH::GetClosestTriangle(const ROOT::Math::XYZVector &point) const
{
   // testpoint object in float for quick BVH interaction
   Vec3 testpoint(point.X(), point.Y(), point.Z());

   // comparator bringing out "smallest" value on top
   auto cmp = [](TBVHInternal::BVHPrioElement a, TBVHInternal::BVHPrioElement b) { return a.value > b.value; };
   TBVHInternal::BVHPrioQueue<decltype(cmp)> queue(cmp);
   queue.clear();

   auto currnode = fBVH->nodes[0]; // we start from the top BVH node
   // algorithm is based on standard iterative tree traversal with priority queues
   auto best_enclosing_R_sq = std::numeric_limits<Double_t>::max();
   auto current_safety_sq = 0.0;
   TGeoTriangleMesh::ClosestTriangle_t closesTGeoTriangle{};
   do {
      if (currnode.is_leaf()) {
         // we are in a leaf node and can now check actual triangle safety distances
         const auto begin_prim_id = currnode.index.first_id();
         const auto end_prim_id = begin_prim_id + currnode.index.prim_count();
         for (auto p_id = begin_prim_id; p_id < end_prim_id; p_id++) {
            const auto triangle_id = fBVH->prim_ids[p_id];
            const TGeoTriangle &triangle = fMesh->TriangleAt(triangle_id);
            const ROOT::Math::XYZVector closestpoint = triangle.ClosestPointToPoint(point);
            const auto safety_sq = (closestpoint - point).Mag2();

            // update best Rmin
            if (safety_sq < best_enclosing_R_sq) {
               best_enclosing_R_sq = safety_sq;
               closesTGeoTriangle.fClosestPoint = closestpoint;
               closesTGeoTriangle.fDistance = sqrt(safety_sq);
               closesTGeoTriangle.fTriangle = &triangle;
               closesTGeoTriangle.fIndex = triangle_id;
            }
         }
      } else {
         // not a leave node ... for further traversal,
         // we inject the children into priority queue based on distance to the child's bounding box
         const auto leftchild_id = currnode.index.first_id();
         const auto rightchild_id = leftchild_id + 1;

         for (size_t childid : {leftchild_id, rightchild_id}) {
            if (childid >= fBVH->nodes.size()) {
               continue;
            }
            const auto &node = fBVH->nodes[childid];
            const auto &thisbbox = node.get_bbox();
            auto inside = TBVHInternal::Contains(thisbbox, testpoint);
            const auto this_safety_sq = inside ? -1.f : TBVHInternal::SafetySqToNode(thisbbox, testpoint);
            if (this_safety_sq <= best_enclosing_R_sq) {
               // this should be further considered
               queue.push(TBVHInternal::BVHPrioElement{childid, this_safety_sq});
            }
         }
      }

      if (queue.size() > 0) {
         auto currElement = queue.top();
         currnode = fBVH->nodes[currElement.bvh_node_id];
         current_safety_sq = currElement.value;
         queue.pop();
      } else {
         break;
      }
   } while (current_safety_sq <= best_enclosing_R_sq);

   return closesTGeoTriangle;
}

////////////////////////////////////////////////////////////////////////////////
/// Get Safety distance. If fAccurateSafety is false, a lower bound approximation
/// for the Safety distance is returned.
///
///\param[in] point
///\return Double_t
///

Double_t TBVH::GetSafetyDistance(const ROOT::Math::XYZVector &point) const
{
   TGeoTriangleMesh::ClosestTriangle_t closesTGeoTriangle = GetClosestTriangle(point);
   return closesTGeoTriangle.fDistance;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the distance to the geometry surface for the ray
/// defined by origin and direction
///
/// \param[in] origin of testray
/// \param[in] direction of testray
/// \param[in] isorigininside indicates if the testray origin lies within the mesh
/// \return Double_t
///

Double_t
TBVH::DistanceInDirection(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction, Bool_t isInside)
{
   // Account for the fact that a point can sit on a triangle. To be able to see the triangle one sits on
   // we have to shift the origin ever so slightly back
   ROOT::Math::XYZVector tmpoffset = direction * -1;
   tmpoffset *= TGeoShape::Tolerance() / TMath::Sqrt(tmpoffset.Mag2());
   ROOT::Math::XYZVector neworigin = (origin + tmpoffset);
   auto ray = Ray{Vec3(neworigin.X(), neworigin.Y(), neworigin.Z()), // Ray origin
                  Vec3(direction.X(), direction.Y(), direction.Z()), // Ray direction
                  0, 1e30};
   const TGeoTriangle *triangle = GetIntersectedTriangle(ray);
   while (triangle != nullptr) {
      if ((isInside && triangle->Normal().Dot(direction) > 0) ||
          (!isInside && triangle->Normal().Dot(direction) <= TGeoShape::Tolerance())) {
         return ray.tmax;
      } else {
         ray = Ray{Vec3(origin.X(), origin.Y(), origin.Z()),          // Ray origin
                   Vec3(direction.X(), direction.Y(), direction.Z()), // Ray direction
                   ray.tmax + TGeoTriangle::sAccuracy,
                   1e30}; // Update minimum tmin to be current tmax for next triangle intersection
         triangle = GetIntersectedTriangle(ray);
      }
   }
   if (isInside)
      return 0;
   // std::cout << "No intersection found" << std::endl;
   return 1e30;
}

////////////////////////////////////////////////////////////////////////////////
/// Find first intersected triangle along the testray
///
/// \param[out] ray along which the next intersected triangle needs to be found
/// \return const TGeoTriangle *

const TGeoTriangle *TBVH::GetIntersectedTriangle(Ray &ray) const
{
   static constexpr size_t stack_size = 64;
   static constexpr bool use_robust_traversal = true;

   const TGeoTriangle *first_intersected_triangle{nullptr};
   // Traverse the BVH and get the u, v coordinates of the closest intersection.
   bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
   fBVH->intersect<false, use_robust_traversal>(ray, fBVH->get_root().index, stack, [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
         size_t triangleId = fBVH->prim_ids[i];
         auto &triangle = fMesh->TriangleAt(triangleId);
         auto t = triangle.DistanceFrom({ray.org[0], ray.org[1], ray.org[2]}, {ray.dir[0], ray.dir[1], ray.dir[2]});
         if (t < ray.tmax && ray.tmin < t) {
            ray.tmax = t;
            first_intersected_triangle = &triangle;
         }
      }
      return false; // Iterate over all triangles
   });

   return first_intersected_triangle;
}

////////////////////////////////////////////////////////////////////////////////
/// Prints the BVH

void TBVH::Print(Option_t *) const
{
   std::cout << "BVH containing " << fBVH->nodes.size() << " nodes" << std::endl;
   for (size_t i = 0; i < fBVH->nodes.size(); ++i) {
      const auto &n = fBVH->nodes[i];
      auto bbox = n.get_bbox();
      auto min = bbox.min;
      auto max = bbox.max;
      long objectid = -1;
      if (n.index.prim_count() > 0) {
         objectid = fBVH->prim_ids[n.index.first_id()];
      }
      std::cout << "NODE id" << i << " "
                << " prim_count: " << n.index.prim_count() << " first_id " << n.index.first_id() << " object_id "
                << objectid << " ( " << min[0] << " , " << min[1] << " , " << min[2] << ")"
                << " ( " << max[0] << " , " << max[1] << " , " << max[2] << ")"
                << "\n";
   }
}

}; // namespace Tessellated