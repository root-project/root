// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TOctree
\ingroup Geometry_classes

Partitioning structure to improve processing time for TGeoTessellated's
navigation functions. Recursively subdivides box into 8 childboxes.
Each box contains list contained triangles. Rather than considering all
triangles in navigation functions, only triangles within relevant boxes are
considered. Navigation functionality for meshes with small triangle
counts suffer from overhead introduced by TOctree, higher triangle count
meshes (> 1000 triangles) can benefit greatly from usage of TOctree.
*/

#include "Tessellated/TOctree.h"

#include <cstdlib>   // for exit
#include <algorithm> // for min, sort, max, copy, set_difference
#include <cmath>     // for abs
#include <iostream>  // for operator<<, basic_ostream, basic_ostream<...
#include <iterator>  // for insert_iterator, begin, end, inserter
#include <limits>    // for numeric_limits
#include <memory>    // for allocator_traits<>::value_type
#include <set>       // for set<>::iterator, set

#include "TGeoShape.h"                    // for TGeoShape
#include "TGeoTessellated.h"              // for TGeoTessellated
#include "Tessellated/TGeoTriangleMesh.h" // for TGeoTriangleMesh::ClosestTriangle_t, TTriang...

namespace Tessellated {

////////////////////////////////////////////////////////////////////////////////
// ClassImp(TOctree);

////////////////////////////////////////////////////////////////////////////////
/// Create a Octree partitioning structure
///
/// \param[in] tsl to create Octree for
/// \param[in] maxdepth of Octree
/// \param[in] maxtriangles per leaf Octant
/// \param[in] accurateSafety should the accurate Safety calculation be used
/// \return std::unique_ptr<TOctree>
std::unique_ptr<TOctree>
TOctree::CreateOctree(const TGeoTessellated *tsl, UInt_t maxdepth, UInt_t maxtriangles, Bool_t accurateSafety)
{

   const Double_t *origin = tsl->GetOrigin();
   const Double_t dx = tsl->GetDX();
   const Double_t dy = tsl->GetDY();
   const Double_t dz = tsl->GetDZ();

   OctreeConfig_t octreeconfig(OctantBounds_t(ROOT::Math::XYZVector(origin[0] - dx, origin[1] - dy, origin[2] - dz),
                                              ROOT::Math::XYZVector(origin[0] + dx, origin[1] + dy, origin[2] + dz)),
                               tsl->GetTriangleMesh(), &tsl->GetUsedTriangleIndices(), maxdepth, maxtriangles, 0);
   octreeconfig.fAccurateSafety = accurateSafety;
   return std::make_unique<TOctree>(octreeconfig);
}

////////////////////////////////////////////////////////////////////////////////
/// Default  Constructor

TOctree::TOctree() : fRoot(nullptr), fIsSetup(false) {}

////////////////////////////////////////////////////////////////////////////////
/// Constructor setting up octree using octree configuration (mesh,
/// octree depth and max triangle in an octant)
TOctree::TOctree(const OctreeConfig_t &octreeconfig) : fRoot(nullptr), fIsSetup(false)
{
   SetupOctree(octreeconfig);
}

////////////////////////////////////////////////////////////////////////////////
/// TOctree destructor

TOctree::~TOctree()
{
   if (fRoot != nullptr) {
      delete fRoot;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set up this octree using octree configuration (mesh,
/// octree depth and max triangle in an octant as well as whether an accurate
/// Safety computation is needed (it is!!)

void TOctree::SetupOctree(const OctreeConfig_t &octreeconfig)
{

   fMesh = octreeconfig.fMesh;
   fAccurateSafety = octreeconfig.fAccurateSafety;
   if (fRoot != nullptr) {
      delete fRoot;
   }
   fRoot = new TOctant(octreeconfig);
   ROOT::Math::XYZVector lowerLeft = octreeconfig.fOctantBounds.fMin;
   ROOT::Math::XYZVector uppercorner = octreeconfig.fOctantBounds.fMax;
   fScale = uppercorner - lowerLeft;
   fMaxLayer = octreeconfig.fMaxDepth;
   fMaxPower = static_cast<UInt_t>(std::pow(2, fMaxLayer));
   std::vector<TOctant const *> octants{};
   GetOctants(fRoot, octants);
   std::set<UInt_t> ids;
   for (const TOctant *octant : octants) {
      std::vector<UInt_t> triangles = octant->GetContainedTriangles();
      ids.insert(triangles.begin(), triangles.end());
   }
   if (ids.size() != fMesh->Triangles().size()) {
      std::cout << "Leaf Octants do not see all triangles. Aborting " << std::endl;
      for (UInt_t i : ids) {
         std::cout << i << std::endl;
      }
      std::exit(-1);
   }
   fIsSetup = true;
}

////////////////////////////////////////////////////////////////////////////////
/// Prints Octree information

void TOctree::Print(Option_t *) const
{
   std::vector<TOctant const *> octants = GetLeafOctants();
   std::cout << "Octree containing " << octants.size() << " leaf octants" << std::endl;
   // for (size_t i = 0; i < octants.size(); ++i) {
   //    const auto *octant = octants[i];
   //    auto min = octant->GetLowerCorner();
   //    auto max = octant->GetUpperCorner();
   //    std::cout << "Octant spanning " << i << " "
   //              << " prim_count: " << n.index.prim_count() << " first_id " << n.index.first_id() << " object_id "
   //              << objectid << " ( " << min[0] << " , " << min[1] << " , " << min[2] << ")"
   //              << " ( " << max[0] << " , " << max[1] << " , " << max[2] << ")"
   //              << "\n";
   // }
}

////////////////////////////////////////////////////////////////////////////////
/// Recursively iterate through each octant and fill the into octants
///
/// \param[in] octant
/// \param[out] octants vector to collect all octants in

void TOctree::GetOctants(TOctant const *octant, std::vector<TOctant const *> &octants) const
{
   if (!octant->IsLeaf()) {
      for (UInt_t i = 0; i < TOctant::sNUMBER_OF_CHILDREN; ++i) {
         GetOctants(octant->GetChild(i), octants);
      }
   } else {
      octants.push_back(octant);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return all leaf octants of the Octree
///
/// \return std::vector<TOctant const *> containing all leaf octants

std::vector<TOctant const *> TOctree::GetLeafOctants() const
{
   std::vector<TOctant const *> octants;
   GetOctants(fRoot, octants);
   return octants;
}

////////////////////////////////////////////////////////////////////////////////
/// Find the leaf octant containing point
///
/// \param[in] point to be located
/// \return  const TOctant * containing point

const TOctant *TOctree::GetRelevantOctant(const ROOT::Math::XYZVector &point) const
{
   const TOctant *octant = fRoot;

   while (octant->IsContainedByOctant(point, TGeoTriangle::sAccuracy) && !octant->IsLeaf()) {
      for (UInt_t i = 0; i < 8; ++i) {
         if (octant->GetChild(i)->IsContainedByOctant(point, TGeoTriangle::sAccuracy)) {
            octant = octant->GetChild(i);
            break;
         }
      }
   }

   return octant;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if point is contained in mesh
///
/// \param[in] point to be tested if it is inside mesh
/// \return Bool_t indicating if point is contained

Bool_t TOctree::IsPointContained(const ROOT::Math::XYZVector &point) const
{
   const TOctant *octant = GetRelevantOctant(point);
   const auto &triIndices = octant->GetContainedTriangles();

   if (!triIndices.empty()) {
      fOrigin = point;
      // ROOT::Math::XYZVector targetPoint = fMesh->TriangleAt(triIndices[0]).Center();
      ROOT::Math::XYZVector dir = {0, 0, 1};
      fDirection = dir.Unit();
      std::vector<TGeoTriangleMesh::IntersectedTriangle_t> triangleIntersections;
      FindClosestFacePoint(point, dir, triangleIntersections);
      if (triangleIntersections.empty()) {
         return false;
      }
      std::sort(std::begin(triangleIntersections), std::end(triangleIntersections));

      return triangleIntersections[0].fDirDotNormal > 0 || triangleIntersections[0].fDistance < TGeoShape::Tolerance();
   } else {
      if (octant->GetState() == TOctant::State::INSIDE) {
         return true;
      }
   }
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Find closest triangle to point
///
/// \param[in] point for which closest triangle needs to be found
/// \return TGeoTriangleMesh::ClosestTriangle_t

TGeoTriangleMesh::ClosestTriangle_t TOctree::GetClosestTriangle(const ROOT::Math::XYZVector &point) const
{

   const TOctant *octant = GetRelevantOctant(point);
   const auto &triIndices = octant->GetContainedTriangles();
   if (!triIndices.empty()) {
      auto closestTriangle = fMesh->FindClosestTriangleInMesh(point, triIndices);
      if (octant->GetMinDistanceToBoundaries(point) + TOctant::sAccuracy > closestTriangle.fDistance) {
         return closestTriangle;
      }
   }
   return GetSafetyDistanceAccurate(point);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the smallest distance between point and mesh.
/// Can return approximation if Set
/// \param[in] point for which smallest distance to mesh needs to be determined
/// \return Double_t for the smallest distance to mesh

Double_t TOctree::GetSafetyDistance(const ROOT::Math::XYZVector &point) const
{
   if (fAccurateSafety) {
      return GetSafetyDistanceAccurate(point).fDistance;
   }
   const TOctant *octant = GetRelevantOctant(point);
   const auto &triIndices = octant->GetContainedTriangles();
   if (!triIndices.empty()) {
      auto closestTriangle = fMesh->FindClosestTriangleInMesh(point, triIndices);
      return std::min(octant->GetMinDistanceToBoundaries(point) + TOctant::sAccuracy, closestTriangle.fDistance);
   }
   return octant->GetMinDistanceToBoundaries(point) + TOctant::sAccuracy;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the smallest accurate distance between point and mesh
///
/// \param[in] point for which smallest distance to mesh needs to be determined
/// \return Double_t for the smallest distance to mesh

TGeoTriangleMesh::ClosestTriangle_t TOctree::GetSafetyDistanceAccurate(const ROOT::Math::XYZVector &point) const
{
   const TOctant *octant = GetRelevantOctant(point);
   const auto &triIndices = octant->GetContainedTriangles();
   TGeoTriangleMesh::ClosestTriangle_t best{};
   if (!triIndices.empty()) {
      auto closestTriangle = fMesh->FindClosestTriangleInMesh(point, triIndices);
      if (closestTriangle.fDistance < octant->GetMinDistanceToBoundaries(point)) {
         return closestTriangle;
      }
      best = closestTriangle;
   } else {
      const auto &triIndices2 = octant->GetParent()->GetContainedTriangles();
      auto closestTriangle = fMesh->FindClosestTriangleInMesh(point, triIndices2);
      if (closestTriangle.fDistance < octant->GetParent()->GetMinDistanceToBoundaries(point)) {
         return closestTriangle;
      }
      best = closestTriangle;
   }
   return GetSafetyInSphere(point, best);
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function to compute the accurate Safety distance for point
/// Determine minimum search radius in which Safety  must lie. This would be the
/// first closest triangle found to point. With that Safety as search radius, all
/// triangles in octants within that sphere have to be tested
///
/// \param[in] point for which smallest distance to mesh needs to be determined
/// \return Double_t for the smallest distance to mesh

TGeoTriangleMesh::ClosestTriangle_t
TOctree::GetSafetyInSphere(const ROOT::Math::XYZVector &point,
                           const TGeoTriangleMesh::ClosestTriangle_t &candidate) const
{
   std::vector<std::pair<TOctant const *, Double_t>> octantDistances;
   FindOctantsInSphere(candidate.fDistance, point, fRoot, octantDistances);
   std::sort(octantDistances.begin(), octantDistances.end(),
             [](const std::pair<TOctant const *, Double_t> &a, const std::pair<TOctant const *, Double_t> &b) {
                return a.second < b.second;
             });
   TGeoTriangleMesh::ClosestTriangle_t best = candidate;
   std::set<UInt_t> seenIndices{};
   for (const auto &octantPair : octantDistances) {

      if (octantPair.second > best.fDistance) {
         return best;
      }
      TOctant const *octant = octantPair.first;
      auto triIndices = octant->GetContainedTriangles();
      std::vector<UInt_t> toBeTested;

      std::set_difference(triIndices.begin(), triIndices.end(), seenIndices.begin(), seenIndices.end(),
                          std::inserter(toBeTested, toBeTested.begin()));
      std::copy(triIndices.begin(), triIndices.end(), std::inserter(seenIndices, seenIndices.end()));

      if (!toBeTested.empty()) {
         auto closesTGeoTriangle = fMesh->FindClosestTriangleInMesh(point, toBeTested);
         if (best.fDistance > closesTGeoTriangle.fDistance) {
            best = closesTGeoTriangle;
         }
      }
   }
   return best;
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function for GetSafetyInSphere to first find all octants in sphere
/// around a test point
///
/// \param[in] radius search radius
/// \param[in] point center around which all octants need to be collected
/// \return Double_t for the smallest distance to mesh

void TOctree::FindOctantsInSphere(Double_t radius, const ROOT::Math::XYZVector &point, TOctant const *octant,
                                  std::vector<std::pair<const TOctant *, Double_t>> &octants) const
{
   for (UInt_t i = 0; i < TOctant::sNUMBER_OF_CHILDREN; ++i) {
      const Double_t distance = octant->GetChild(i)->GetMinDistance(point);
      if (distance <= radius) {
         if (octant->GetChild(i)->IsLeaf()) {
            octants.emplace_back(octant->GetChild(i), distance);
         } else {
            FindOctantsInSphere(radius, point, octant->GetChild(i), octants);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function for DistanceInDirection for when testpoint is inside of
/// the mesh

Double_t TOctree::DistFromInside(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction,
                                 Bool_t isorigininside,
                                 const std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections)
{
   size_t size = triangleIntersections.size();
   size_t counter = 0;
   if (size == 0) {
      return 0;
   }
   while (counter < size) {
      if (triangleIntersections[counter].fDistance < TGeoShape::Tolerance()) {
         ++counter;
         // return triangleIntersections[counter].fDistance;
      } else if (triangleIntersections[counter].fDirDotNormal < 0) {
         ++counter;
      } else {
         return triangleIntersections[counter].fDistance;
      }
   }
   // if we reach here, there was no intersection in direction with a triangle with a facenormal parallel to the
   // direction. So we could be actually outside of the geometry or on a triangle looking out.
   // Test that the closest triangles in direction to the direction indicate this
   if (triangleIntersections[0].fDistance < TGeoShape::Tolerance() || triangleIntersections[0].fDirDotNormal > 0) {
      return 0.0;
   }

   // if point lies inside triangle mesh, one must find a triangle with a facenormal pointing away
   // Hence, the origin is slightly shifted and the ray is reshot
   ROOT::Math::XYZVector orthogonal = Tessellated::XYZVectorHelper::Orthogonal(direction);
   Double_t smallestExtent = std::min(fScale.X(), std::min(fScale.Y(), fScale.Z()));
   orthogonal *= (0.01 * smallestExtent) / TMath::Sqrt(orthogonal.Mag2());

   ROOT::Math::XYZVector npointa = {origin.X() - orthogonal.X(), origin.Y() - orthogonal.Y(),
                                    origin.Z() - orthogonal.Z()};
   std::cerr
      << "TOctree::DistanceInDirection((" << origin.X() << "," << origin.Y() << ", " << origin.Z() << "),"
      << "(" << direction.X() << "," << direction.Y() << ", " << direction.Z() << "),...) from the inside found "
      << " triangles in direction, or all triangles are parallel to direction (even though we are in the geometry)"
      << " -> We must be hitting the edge of two triangles. We reshoot from a slightly moved point" << std::endl;
   return DistanceInDirection(npointa, direction, isorigininside);
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function for DistanceInDirection for when testpoint is outside of
/// the mesh

Double_t TOctree::DistFromOutside(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction,
                                  Bool_t isorigininside,
                                  const std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections)
{
   size_t size = triangleIntersections.size();
   size_t counter = 0;

   // Bool_t noDistance = kTRUE;
   // Double_t distance = 0;
   while (counter < size) {
      // If you sit on the triangle, ignore it
      if (triangleIntersections[counter].fDistance < TGeoShape::Tolerance()) {
         ++counter;
         // return triangleIntersections[counter].fDistance;
      } else if (triangleIntersections[counter].fDirDotNormal > TGeoTriangle::sAccuracy /*&& (noDistance || std::abs(triangleIntersections[counter].fDistance-distance) < TGeoShape::Tolerance())*/) {
         // distance = triangleIntersections[counter].fDistance;
         // noDistance = kFALSE;
         ++counter;
      } else {
         return triangleIntersections[counter].fDistance;
      }
   }

   if (size > 0 && !IsPointContained(origin)) {
      ROOT::Math::XYZVector orthogonal = Tessellated::XYZVectorHelper::Orthogonal(direction);
      Double_t smallestExtent = std::min(fScale.X(), std::min(fScale.Y(), fScale.Z()));
      orthogonal *= (0.01 * smallestExtent) / TMath::Sqrt(orthogonal.Mag2());
      ROOT::Math::XYZVector npointa = {origin.X() - orthogonal.X(), origin.Y() - orthogonal.Y(),
                                       origin.Z() - orthogonal.Z()};
      std::cerr << "TOctree::DistanceInDirection((" << origin.X() << "," << origin.Y() << ", " << origin.Z() << "),"
                << "(" << direction.X() << "," << direction.Y() << ", " << direction.Z()
                << "),...) from the outside found " << std::endl;
      return DistanceInDirection(npointa, direction, isorigininside);
   }
   return 1e30;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the distance to the geometry surface for the ray defined by origin and direction
///
/// \param[in] origin
/// \param[in] direction
/// \param[in] isorigininside
/// \return Double_t

Double_t TOctree::DistanceInDirection(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction,
                                      Bool_t isorigininside)
{
   fOrigin = origin;
   fDirection = direction.Unit();
   fOriginInside = isorigininside;
   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> triangleIntersections;
   FindClosestFacePoint(fOrigin, fDirection, triangleIntersections);
   std::sort(std::begin(triangleIntersections), std::end(triangleIntersections));
   if (isorigininside) {
      return DistFromInside(fOrigin, fDirection, isorigininside, triangleIntersections);
   }
   return DistFromOutside(fOrigin, fDirection, isorigininside, triangleIntersections);
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function for DistanceInDirection -> FindClosestFacePoint to find all
/// triangles intersected by a test ray given in fOrigin/fDirection
/// For a given octant find all triangle intersections within it with ray
/// described by fOrigin and fDirection
Bool_t TOctree::CheckFacesInOctant(const TOctant *octant,
                                   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections) const
{
   const std::vector<UInt_t> &faceids = octant->GetContainedTriangles();
   Bool_t FoundRelevant = kFALSE;

   for (UInt_t index : faceids) {
      const TGeoTriangle &triangle = fMesh->TriangleAt(index);
      const double currentdistance = triangle.DistanceFrom(fOrigin, fDirection);
      if (currentdistance > -TGeoTriangle::sAccuracy) {
         const ROOT::Math::XYZVector currentIntersectionPoint = fOrigin + currentdistance * fDirection;
         if (octant->IsContainedByOctant(currentIntersectionPoint, TGeoTriangle::sAccuracy)) {
            const double dot = triangle.Normal().Dot(fDirection);
            triangleIntersections.push_back(TGeoTriangleMesh::IntersectedTriangle_t{
               &triangle, index, currentIntersectionPoint, currentdistance, dot});
            if ((fOriginInside && dot > 0) || (!fOriginInside && dot < TGeoTriangle::sAccuracy)) {
               FoundRelevant = kTRUE;
            }
         }
      }
   }

   return FoundRelevant;
}

////////////////////////////////////////////////////////////////////////////////
// Implementation following "An Efficient Parametric Algorithm for Octree Traversal",
//            written by J.Revelles, C.Urena and M.Lastra
//
//  title={An Efficient Parametric Algorithm for Octree Traversal},
//  author={J. Revelles and Carlos Ure{\~n}a and Miguel Lastra},
//  booktitle={WSCG},
//  year={2000}
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Helper function for Fast Octree Traversal

Int_t TOctree::FindFirstNodeIndex(Double_t tx0, Double_t ty0, Double_t tz0, Double_t txM, Double_t tyM,
                                  Double_t tzM) const
{
   Int_t index = 0;
   // select the entry plane and determine the index of the first intersected child of the next mothernode
   // entry plane is plane perpendicular to ray component with largest time (max(tx0,ty0,tz0))

   // Entry Plane XY
   if (tz0 > tx0) {
      if (tz0 > ty0) {
         if (txM < tz0) {
            index |= 4;
         }
         if (tyM < tz0) {
            index |= 2;
         }
         return index;
      }
   }
   // Entry Plane YZ
   else {
      if (tx0 > ty0) {
         if (tyM < tx0) {
            index |= 2;
         }
         if (tzM < tx0) {
            index |= 1;
         }
         return index;
      }
   }
   // Entry Plane XZ
   if (txM < ty0) {
      index |= 4;
   }
   if (tzM < ty0) {
      index |= 1;
   }
   return index;
}

////////////////////////////////////////////////////////////////////////////////
/// The minimum of tx0(octant), ty0(octant) and tz0(octant) (here named t...M)
/// determine, which sisternode of the current octant is beeing entered after
/// the ray passed through current octant. The next nodes index is passed in as
/// x,y, and z
Int_t TOctree::FindNextNodeIndex(Double_t txM, Int_t x, Double_t tyM, Int_t y, Double_t tzM, Int_t z) const
{
   if (txM < tyM) {
      if (txM < tzM) {
         return x;
      }
   } else {
      if (tyM < tzM) {
         return y;
      }
   }
   return z;
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function for FindClosestFacePoint (itself helper to
/// DistanceInDirection)

Bool_t TOctree::ProcessSubtree(Double_t tx0, Double_t ty0, Double_t tz0, Double_t tx1, Double_t ty1, Double_t tz1,
                               const TOctant *octant, const ROOT::Math::XYZVector &origin,
                               const ROOT::Math::XYZVector &direction,
                               std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections) const
{

   Bool_t result = false;
   if ((tx1 < 0) || (ty1 < 0) || (tz1 < 0)) {
      return result;
   }
   if (octant->IsLeaf()) {
      if (octant->GetContainedTriangles().empty()) {
         return result;
      }

      return CheckFacesInOctant(octant, triangleIntersections);
   }

   Double_t txM = 0.5 * (tx0 + tx1);
   Double_t tyM = 0.5 * (ty0 + ty1);
   Double_t tzM = 0.5 * (tz0 + tz1);

   // handle rays with zero direction component
   Double_t max = std::numeric_limits<Double_t>::max();
   const ROOT::Math::XYZVector &lower = octant->GetLowerCorner();
   const ROOT::Math::XYZVector &upper = octant->GetUpperCorner();
   if (std::abs(direction.X()) < fTolerance) {
      txM = -max;
      if (origin.X() < (lower.X() + upper.X()) * 0.5) {
         txM = max;
      }
   }
   if (std::abs(direction.Y()) < fTolerance) {
      tyM = -max;
      if (origin.Y() < (lower.Y() + upper.Y()) * 0.5) {
         tyM = max;
      }
   }
   if (std::abs(direction.Z()) < fTolerance) {
      tzM = -max;
      if (origin.Z() < (lower.Z() + upper.Z()) * 0.5) {
         tzM = max;
      }
   }

   Int_t currNode = FindFirstNodeIndex(tx0, ty0, tz0, txM, tyM, tzM);

   do {

      switch (currNode) {
      case 0: {
         result = ProcessSubtree(tx0, ty0, tz0, txM, tyM, tzM, octant->GetChild(fIndexByte), origin, direction,
                                 triangleIntersections);
         currNode = FindNextNodeIndex(txM, 4, tyM, 2, tzM, 1);
         break;
      }
      case 1: {
         result = ProcessSubtree(tx0, ty0, tzM, txM, tyM, tz1, octant->GetChild(1 ^ fIndexByte), origin, direction,
                                 triangleIntersections);
         currNode = FindNextNodeIndex(txM, 5, tyM, 3, tz1, 8);
         break;
      }
      case 2: {
         result = ProcessSubtree(tx0, tyM, tz0, txM, ty1, tzM, octant->GetChild(2 ^ fIndexByte), origin, direction,
                                 triangleIntersections);
         currNode = FindNextNodeIndex(txM, 6, ty1, 8, tzM, 3);
         break;
      }
      case 3: {
         result = ProcessSubtree(tx0, tyM, tzM, txM, ty1, tz1, octant->GetChild(3 ^ fIndexByte), origin, direction,
                                 triangleIntersections);
         currNode = FindNextNodeIndex(txM, 7, ty1, 8, tz1, 8);
         break;
      }
      case 4: {
         result = ProcessSubtree(txM, ty0, tz0, tx1, tyM, tzM, octant->GetChild(4 ^ fIndexByte), origin, direction,
                                 triangleIntersections);
         currNode = FindNextNodeIndex(tx1, 8, tyM, 6, tzM, 5);
         break;
      }
      case 5: {
         result = ProcessSubtree(txM, ty0, tzM, tx1, tyM, tz1, octant->GetChild(5 ^ fIndexByte), origin, direction,
                                 triangleIntersections);
         currNode = FindNextNodeIndex(tx1, 8, tyM, 7, tz1, 8);
         break;
      }
      case 6: {
         result = ProcessSubtree(txM, tyM, tz0, tx1, ty1, tzM, octant->GetChild(6 ^ fIndexByte), origin, direction,
                                 triangleIntersections);
         currNode = FindNextNodeIndex(tx1, 8, ty1, 8, tzM, 7);
         break;
      }
      case 7: {
         result = ProcessSubtree(txM, tyM, tzM, tx1, ty1, tz1, octant->GetChild(7 ^ fIndexByte), origin, direction,
                                 triangleIntersections);
         currNode = 8;
         break;
      }
      }
   } while (currNode < 8 && !result);

   return result;
}

////////////////////////////////////////////////////////////////////////////////
Bool_t TOctree::FindClosestFacePoint(ROOT::Math::XYZVector origin, ROOT::Math::XYZVector direction,
                                     std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections) const
{

   fIndexByte = 0;

   const ROOT::Math::XYZVector &upper = fRoot->GetUpperCorner();
   const ROOT::Math::XYZVector &lower = fRoot->GetLowerCorner();
   // ROOT::Math::XYZVector scale = (upper - lower);
   const ROOT::Math::XYZVector center = (upper + lower) * 0.5;
   direction = direction.Unit();

   //"transform" negative direction components by flipping the direction components sign and  the origin component
   // around the center and take the flip into account for the first
   // intersected octant by masking with the fIndexByte
   if (direction.X() < 0) {
      origin.SetX(center.X() + (center.X() - origin.X()));
      direction.SetX(-direction.X());
      fIndexByte |= 4;
   }

   if (direction.Y() < 0) {
      origin.SetY(center.Y() + (center.Y() - origin.Y()));
      direction.SetY(-direction.Y());
      fIndexByte |= 2;
   }

   if (direction.Z() < 0) {
      origin.SetZ(center.Z() + (center.Z() - origin.Z()));
      direction.SetZ(-direction.Z());
      fIndexByte |= 1;
   }

   const Double_t invx = 1 / direction.X();
   const Double_t invy = 1 / direction.Y();
   const Double_t invz = 1 / direction.Z();

   const Double_t tx0 = (lower.X() - origin.X()) * invx;
   const Double_t tx1 = (upper.X() - origin.X()) * invx;
   const Double_t ty0 = (lower.Y() - origin.Y()) * invy;
   const Double_t ty1 = (upper.Y() - origin.Y()) * invy;
   const Double_t tz0 = (lower.Z() - origin.Z()) * invz;
   const Double_t tz1 = (upper.Z() - origin.Z()) * invz;
   if (std::max({tx0, ty0, tz0}) < std::min({tx1, ty1, tz1})) {
      return ProcessSubtree(tx0, ty0, tz0, tx1, ty1, tz1, fRoot, origin, direction, triangleIntersections);
   }

   return kFALSE;
}

}; // namespace Tessellated
