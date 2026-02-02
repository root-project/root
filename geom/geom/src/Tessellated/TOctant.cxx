// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TOctant
\ingroup Geometry_classes

Helper class for Tessellated::TOctree. TOctant represents box containing list
of triangle indices that it contains. Implements functionality to recursively
subdivide itself, creating its own child boxes, and functionality to detect
triangle box intersections, the be able to determine the triangles contained by itself.
*/

#include "Tessellated/TOctant.h"

#include <cmath>     // for abs
#include <algorithm> // for min, max
#include <iostream>  // for operator<<, endl, basic_ostream, cout
#include <iterator>  // for begin, end
#include <memory>    // for allocator_traits<>::value_type
#include <numeric>   // for iota

#include "TGeoShape.h"                    // for TGeoShape
#include "Tessellated/TGeoTriangle.h"     // for TGeoTriangle
#include "Tessellated/TGeoTriangleMesh.h" // for TGeoTriangleMesh::IntersectedTriangle_t, TTr...

namespace Tessellated {

unsigned int TOctant::sNumberOfInsideOctants = 0;
unsigned int TOctant::sNumberOfOutsideOctants = 0;
unsigned int TOctant::sNumberOfLeafOctants = 0;
const Double_t TOctant::sAccuracy = 1.5E-6;

// ClassImp(TOctant)

////////////////////////////////////////////////////////////////////////////////
/// Default constructor
TOctant::TOctant()
{
   InitializeChildren();
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor setting up Octant to create intended octree
///
/// \param[in] octantconfig describing depth and triangle count for octree

TOctant::TOctant(const OctreeConfig_t &octantconfig)
{
   InitializeChildren();
   SetupOctant(octantconfig);
   CreateChildOctants(octantconfig);
   SetState(octantconfig.fMesh);

   // remove the ids to save memory if the octant is merely an intermediate octant
   //(neither a leaf itself, nor the parent of a leaf, nor the Root).
   Bool_t needsTriangleIds = (IsLeaf() || fParent == nullptr);
   if (!needsTriangleIds) {
      needsTriangleIds =
         std::any_of(std::begin(fChildren), std::end(fChildren), [](const TOctant *a) { return a->IsLeaf(); });
   }
   if (!needsTriangleIds) {
      fContainedTriangles.clear();
      fContainedTriangles.resize(0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// TOctant destructor

TOctant::~TOctant()
{
   for (const auto *child : fChildren) {
      if (child != nullptr) {
         delete child;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize all children pointer to be nullptr to start with

void TOctant::InitializeChildren()
{
   std::fill(std::begin(fChildren), std::end(fChildren), nullptr);
}

////////////////////////////////////////////////////////////////////////////////
/// Setup octant size and position in overall octree as well as which triangles
/// are contained by this octant

void TOctant::SetupOctant(const OctreeConfig_t &octantconfig)
{
   fMin = octantconfig.fOctantBounds.fMin;
   fMax = octantconfig.fOctantBounds.fMax;
   fContainedTriangles = *octantconfig.fContainedTriangles;
   fCurrentTreeDepth = octantconfig.fCurrentTreeDepth;
}

////////////////////////////////////////////////////////////////////////////////
/// Return number of octants counting  itself
/// \return number of octants

UInt_t TOctant::GetNumberOfOctants() const
{
   UInt_t result{1U}; // start with one for the root octant
   for (const auto *child : fChildren) {
      if (child != nullptr) {
         result += child->GetNumberOfOctants();
      }
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the spatial boundaries for the 8 sub-boxes contained by this octant
/// \return std::vector<OctantBounds_t>

std::vector<OctantBounds_t> TOctant::CreateChildBounds(const OctantBounds_t &bounds) const
{
   // Determine new edge vertices of daughter voxels.
   std::vector<OctantBounds_t> newcorners;
   const ROOT::Math::XYZVector &min = bounds.fMin;
   const ROOT::Math::XYZVector &max = bounds.fMax;
   ROOT::Math::XYZVector middlepoint = (min + (max - min) * 0.5);
   ROOT::Math::XYZVector low{0, 0, 0};
   ROOT::Math::XYZVector up{0, 0, 0};

   for (UInt_t i = 0; i < sNUMBER_OF_CHILDREN; ++i) {
      low.SetXYZ(middlepoint.X(), middlepoint.Y(), middlepoint.Z());
      up.SetXYZ(max.X(), max.Y(), max.Z());

      if (i < 4) {
         low.SetX(min.X());
         up.SetX(middlepoint.X());
      }
      if (i % 2 == 0) {
         low.SetZ(min.Z());
         up.SetZ(middlepoint.Z());
      }
      if (i == 0 || i == 1 || i == 4 || i == 5) {
         low.SetY(min.Y());
         up.SetY(middlepoint.Y());
      }
      newcorners.emplace_back(OctantBounds_t{low, up});
   }
   return newcorners;
}
////////////////////////////////////////////////////////////////////////////////
/// Implementation of AABB-triangle overlap test following
/// Akenine-Möller, Tomas. (2004). Fast 3D Triangle-Box Overlap Testing. Journal of Graphics
/// Tools. 6. 10.1145/1198555.1198747.
////////////////////////////////////////////////////////////////////////////////
/// Helper function to determine if a triangle intersects with an box

Bool_t TOctant::IsNormalAxisSeparating(const ThreeVector3s_t &triVertices, Int_t component,
                                       const ROOT::Math::XYZVector &extents) const
{
   // Since the XYZVector does not seem to define GetCoordinate(index) this becomes rather ... questionable
   const Double_t c1{Tessellated::XYZVectorHelper::ToArray(triVertices.vec1)[component]};
   const Double_t c2{Tessellated::XYZVectorHelper::ToArray(triVertices.vec2)[component]};
   const Double_t c3{Tessellated::XYZVectorHelper::ToArray(triVertices.vec3)[component]};
   const Double_t ec{Tessellated::XYZVectorHelper::ToArray(extents)[component]};
   return std::max({c1, c2, c3}) < -ec - TGeoShape::Tolerance() || std::min({c1, c2, c3}) > ec + TGeoShape::Tolerance();
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function to determine if a triangle intersects with an box

Bool_t TOctant::IsSeparatingAxis(const ThreeVector3s_t &triVertices, const ROOT::Math::XYZVector &axis,
                                 const ROOT::Math::XYZVector & /*edge*/, const ROOT::Math::XYZVector &extents) const
{
   const Double_t p0 = axis.Dot(triVertices.vec1);
   const Double_t p1 = axis.Dot(triVertices.vec2);
   const Double_t p2 = axis.Dot(triVertices.vec3);

   const Double_t r =
      extents.X() * std::abs(axis.X()) + extents.Y() * std::abs(axis.Y()) + extents.Z() * std::abs(axis.Z());
   // Double_t r = extents.X() * std::abs(edge.X()) + extents.Y() * std::abs(edge.Y()) + extents.Z() *
   // std::abs(edge.Z()); Double_t testval = std::max(-std::max(p0, std::max(p1, p2)), std::min(p0, std::min(p1, p2)));

   return std::max({p0, p1, p2}) < -r - TGeoShape::Tolerance() || std::min({p0, p1, p2}) > r + TGeoShape::Tolerance();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute if a triangle intersects with the octant

Bool_t TOctant::TriangleOctantBoundsIntersection(const TGeoTriangle &triangle, const OctantBounds_t &octantbounds) const
{

   ROOT::Math::XYZVector octcenter = ((octantbounds.fMax + octantbounds.fMin) * 0.5);
   ROOT::Math::XYZVector extents = octantbounds.fMax - octcenter;
   extents += ROOT::Math::XYZVector{TOctant::sAccuracy, TOctant::sAccuracy, TOctant::sAccuracy};
   ThreeVector3s_t trivertices;
   trivertices.vec1 = (triangle.Point(0) - octcenter);
   trivertices.vec2 = (triangle.Point(1) - octcenter);
   trivertices.vec3 = (triangle.Point(2) - octcenter);

   // Compute edge vectors for triangle
   ROOT::Math::XYZVector edge10 = (trivertices.vec2 - trivertices.vec1);
   ROOT::Math::XYZVector edge21 = (trivertices.vec3 - trivertices.vec2);
   ROOT::Math::XYZVector edge02 = (trivertices.vec1 - trivertices.vec3);

   // Compute edge vectors for triangle
   // ROOT::Math::XYZVector edge10 = (triangle.Point(1) - triangle.Point(0));
   // ROOT::Math::XYZVector edge21 = (triangle.Point(2) - triangle.Point(1));
   // ROOT::Math::XYZVector edge02 = (triangle.Point(0) - triangle.Point(2));

   // Compute each axis perpendicular to  one of the unit-axis vectors and triangle edge

   auto a00 = ROOT::Math::XYZVector(0, -edge10.Z(), edge10.Y()).Unit(); // edge10 cross e_x
   auto a01 = ROOT::Math::XYZVector(0, -edge21.Z(), edge21.Y()).Unit();
   auto a02 = ROOT::Math::XYZVector(0, -edge02.Z(), edge02.Y()).Unit();
   auto a10 = ROOT::Math::XYZVector(edge10.Z(), 0, -edge10.X()).Unit();
   auto a11 = ROOT::Math::XYZVector(edge21.Z(), 0, -edge21.X()).Unit();
   auto a12 = ROOT::Math::XYZVector(edge02.Z(), 0, -edge02.X()).Unit();
   auto a20 = ROOT::Math::XYZVector(-edge10.Y(), edge10.X(), 0).Unit();
   auto a21 = ROOT::Math::XYZVector(-edge21.Y(), edge21.X(), 0).Unit();
   auto a22 = ROOT::Math::XYZVector(-edge02.Y(), edge02.X(), 0).Unit();

   // Test axis
   if (IsSeparatingAxis(trivertices, a00, edge10, extents)) {
      return false;
   }
   if (IsSeparatingAxis(trivertices, a01, edge21, extents)) {
      return false;
   }
   if (IsSeparatingAxis(trivertices, a02, edge02, extents)) {
      return false;
   }
   if (IsSeparatingAxis(trivertices, a10, edge10, extents)) {
      return false;
   }
   if (IsSeparatingAxis(trivertices, a11, edge21, extents)) {
      return false;
   }
   if (IsSeparatingAxis(trivertices, a12, edge02, extents)) {
      return false;
   }
   if (IsSeparatingAxis(trivertices, a20, edge10, extents)) {
      return false;
   }
   if (IsSeparatingAxis(trivertices, a21, edge21, extents)) {
      return false;
   }
   if (IsSeparatingAxis(trivertices, a22, edge02, extents)) {
      return false;
   }

   if (IsNormalAxisSeparating(trivertices, 0, extents)) {
      return false;
   }
   if (IsNormalAxisSeparating(trivertices, 1, extents)) {
      return false;
   }
   if (IsNormalAxisSeparating(trivertices, 2, extents)) {
      return false;
   }

   ROOT::Math::XYZVector nNormal = edge10.Cross(edge21);
   nNormal = nNormal.Unit();

   Double_t r =
      extents.X() * std::abs(nNormal.X()) + extents.Y() * std::abs(nNormal.Y()) + extents.Z() * std::abs(nNormal.Z());

   Double_t boxCenterDistance = nNormal.Dot(trivertices.vec1);

   return std::abs(boxCenterDistance) <= r + TGeoShape::Tolerance();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the indices of the triangles contained by the provided octant bounds
/// \return std::vector<UInt_t>

std::vector<UInt_t> TOctant::ContainedTriangles(const OctantBounds_t &octant,
                                                const std::vector<UInt_t> &parentstriangles,
                                                const TGeoTriangleMesh *mesh) const
{
   std::vector<UInt_t> result;
   for (UInt_t index : parentstriangles) {
      if (TriangleOctantBoundsIntersection(mesh->TriangleAt(index), octant)) {
         result.push_back(index);
      }
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Precompute whether this octant is fully inside the mesh, fully outside the mesh
/// or if it is mixed, meaning it contains a triangle
/// Knowing that the octant is fully inside/outside can help with the
/// TGeoTessellated::Contains computation.

void TOctant::SetState(const TGeoTriangleMesh *mesh)
{
   if (!fContainedTriangles.empty()) {
      fState = State::MIXED;
      return;
   }
   ROOT::Math::XYZVector point = (fMin + fMax) * 0.5;
   ROOT::Math::XYZVector dir{0.0, 0.0, 1.0};
   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> indir{};
   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> oppdir{};
   std::vector<UInt_t> indices(mesh->Triangles().size());
   std::iota(indices.begin(), indices.end(), 0);
   mesh->FindClosestIntersectedTriangles(point, dir, indices, indir, oppdir);
   if ((indir.size() % 2 == 1 && oppdir.empty()) || (oppdir.size() % 2 == 1 && indir.empty())) {
      std::cout << "TRIANGLE EDGE ISSUE!" << std::endl;
      indir.clear();
      oppdir.clear();
      dir = mesh->TriangleAt(0).Center(); // something went wrong, an edge was hit and not properly recognized
      dir = dir.Unit();
      mesh->FindClosestIntersectedTriangles(point, dir, indices, indir, oppdir);
   }
   if (indir.empty() && oppdir.empty()) {
      fState = State::OUTSIDE;
      ++sNumberOfOutsideOctants;
      return;
   }
   if (!indir.empty() && !oppdir.empty()) {
      if (indir[0].fDirDotNormal > 0 && oppdir[0].fDirDotNormal < 0) {
         fState = State::INSIDE;
         ++sNumberOfInsideOctants;
         return;
      }
   }
   fState = State::OUTSIDE;
}

////////////////////////////////////////////////////////////////////////////////
/// Create all 8 child octants, setting up their boundaries and computing for
/// each, which triangles are contained by each

void TOctant::CreateChildOctants(const OctreeConfig_t &octantconfig)
{

   if (fCurrentTreeDepth + 1 > octantconfig.fMaxDepth || fContainedTriangles.size() <= octantconfig.fMaxTriangles) {
      ++sNumberOfLeafOctants;
      return;
   }

   auto bounds = CreateChildBounds(OctantBounds_t(fMin, fMax));
   UInt_t childindex = 0;
   for (const OctantBounds_t &bound : bounds) {
      ROOT::Math::XYZVector octcenter = ((bound.fMax + bound.fMin) * 0.5);
      ROOT::Math::XYZVector extents = bound.fMax - octcenter;

      auto containedtriangles = ContainedTriangles(bound, *octantconfig.fContainedTriangles, octantconfig.fMesh);

      OctreeConfig_t childconfig;
      childconfig.fOctantBounds = bound;
      childconfig.fMesh = octantconfig.fMesh;
      childconfig.fContainedTriangles = &containedtriangles;
      childconfig.fCurrentTreeDepth = octantconfig.fCurrentTreeDepth + 1;
      childconfig.fMaxDepth = octantconfig.fMaxDepth;
      childconfig.fMaxTriangles = octantconfig.fMaxTriangles;

      fChildren[childindex] = new TOctant(childconfig);
      fChildren[childindex]->SetParent(this);

      ++childindex;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Test if a point lies inside this octant
///
/// \param[in] point to be tested
/// \param[in] epsilon for floating point precision
/// \return Bool_t whether point is contained or not

Bool_t TOctant::IsContainedByOctant(const ROOT::Math::XYZVector &point, Double_t epsilon) const
{
   if (point.X() + epsilon < fMin.X() || point.X() - epsilon > fMax.X()) {
      return false;
   } else if (point.Y() + epsilon < fMin.Y() || point.Y() - epsilon > fMax.Y()) {
      return false;
   } else if (point.Z() + epsilon < fMin.Z() || point.Z() - epsilon > fMax.Z()) {
      return false;
   }
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Find minimum distance of point to boundaries of the octant
/// Used to be used for faster Safety approximation, which turned out to be
/// errornous
///
/// \param[in] point for which distance should be calculated
/// \return Double_t shortest distance of point to octant boundaries

Double_t TOctant::GetMinDistanceToBoundaries(const ROOT::Math::XYZVector &point) const
{
   ROOT::Math::XYZVector diffpointlower = (fMin)-point;
   ROOT::Math::XYZVector diffpointupper = (fMax)-point;

   // get the smallest distance component of box to point
   Double_t min1 =
      std::min(std::abs(diffpointlower.X()), std::min(std::abs(diffpointlower.Y()), std::abs(diffpointlower.Z())));
   Double_t min2 =
      std::min(std::abs(diffpointupper.X()), std::min(std::abs(diffpointupper.Y()), std::abs(diffpointupper.Z())));

   return std::min(min1, min2);
}

////////////////////////////////////////////////////////////////////////////////
/// Print octant information to stdout

void TOctant::Print(Option_t * /*option*/) const
{
   std::cout << "Octant at level " << fCurrentTreeDepth << std::endl;
   Tessellated::XYZVectorHelper::Print(fMin);
   Tessellated::XYZVectorHelper::Print(fMax);
   std::cout << "Contains " << fContainedTriangles.size() << " Triangles" << std::endl;
   for (UInt_t id : fContainedTriangles) {
      std::cout << "ID:" << id << " ";
   }
   std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Find minimum distance of point to triangles within octant or octant boundaries
///
/// \param[in] point for which distance should be calculated
/// \return Double_t shortest distance of point to triangles/octant boundaries

Double_t TOctant::GetMinDistance(const ROOT::Math::XYZVector &point) const
{
   ROOT::Math::XYZVector mid = (fMax + fMin) * (1. / 2.);
   ROOT::Math::XYZVector relPoint = point - mid;
   relPoint.SetXYZ(std::abs(relPoint.X()), std::abs(relPoint.Y()), std::abs(relPoint.Z()));

   ROOT::Math::XYZVector ext = mid - fMin;
   ROOT::Math::XYZVector distanceVec = relPoint - ext;
   distanceVec.SetXYZ(((distanceVec.X() < 0) ? 0 : distanceVec.X()), ((distanceVec.Y() < 0) ? 0 : distanceVec.Y()),
                      ((distanceVec.Z() < 0) ? 0 : distanceVec.Z()));
   Double_t max = distanceVec.X();
   // max = (max < distanceVec.X()) ? distanceVec.X() : max;
   max = (max < distanceVec.Y()) ? distanceVec.Y() : max;
   max = (max < distanceVec.Z()) ? distanceVec.Z() : max;

   if (max <= 0) {
      return 0;
   }
   return Tessellated::XYZVectorHelper::Mag(distanceVec);
}

}; // namespace Tessellated