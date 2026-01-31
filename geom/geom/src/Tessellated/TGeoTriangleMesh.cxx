// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoTriangleMesh
\ingroup Geometry_classes

Helper class for TGeoTessellated. Holds a set of triangles and vertices (TVector3)
and is provides functions to return intersected triangles with ray or finds closest
triangle to point. Includes further functionality to naively test the contained mesh
for closure.
*/

#include "Tessellated/TGeoTriangleMesh.h"

#include <cmath>     // for abs
#include <algorithm> // for max, sort, copy
#include <array>     // for array
#include <iostream>  // for cout, endl
#include <iterator>  // for begin, end
#include <memory>    // for allocator_traits<>::value_type

#include "TBuffer.h"   // for TBuffer
#include "TClass.h"    // for TClass
#include "Tessellated/TGeoTriangle.h" // for TGeoTriangle, TGeoTriangle::ClosestPoint_t, TTriang...

namespace Tessellated {

///////////////////////////////////////////////////////////////////////////////
// ClassImp(TGeoTriangleMesh);

///////////////////////////////////////////////////////////////////////////////
/// TGeoTriangleMesh default constructor

TGeoTriangleMesh::TGeoTriangleMesh()
   : TObject(), fPoints(std::vector<TVector3>(0)), fTriangles(std::vector<TGeoTriangle>(0)), fMeshFile("")
{
} /* = default; */

///////////////////////////////////////////////////////////////////////////////
/// TGeoTriangleMesh constructor setting the meshfilename

TGeoTriangleMesh::TGeoTriangleMesh(const TString &meshfile)
   : TObject(), fPoints(std::vector<TVector3>(0)), fTriangles(std::vector<TGeoTriangle>(0)), fMeshFile(meshfile)
{
} /* = default; */

///////////////////////////////////////////////////////////////////////////////
/// TGeoTriangleMesh destructor

TGeoTriangleMesh::~TGeoTriangleMesh() {} /* = default; */

///////////////////////////////////////////////////////////////////////////////
/// Get the Triangle Indices
///
/// \return  std::vector<UInt_t>

std::vector<UInt_t> TGeoTriangleMesh::GetTriangleIndices() const
{
   std::vector<UInt_t> triangleIndices;
   triangleIndices.resize(fTriangles.size());
   std::iota(triangleIndices.begin(), triangleIndices.end(), 0);
   return triangleIndices;
}

////////////////////////////////////////////////////////////////////////////////
/// Determine first triangle intersected by ray formed by origin,
/// direction out of the pool of allowed triangles usedTriangleIndeices and
/// fill the intersected triangles into indirection or againsdirection
///
/// @param[in] origin
/// @param[in] direction
/// @param[in] usedTriangleIndices
/// @param[out] indirection
/// @param[out] againstdirection

void TGeoTriangleMesh::FindClosestIntersectedTriangles(const TVector3 &origin, const TVector3 &direction,
                                                    const std::vector<UInt_t> &usedTriangleIndices,
                                                    std::vector<IntersectedTriangle_t> &indirection, 
                                                    std::vector<IntersectedTriangle_t> &oppdirection) const
{
   for (UInt_t index : usedTriangleIndices) {
      const TGeoTriangle &triangle = fTriangles[index];
      const auto t = triangle.DistanceFrom(origin, direction);
      if (t != TGeoTriangle::sINF) {
         if (t > 0) {
            indirection.push_back(IntersectedTriangle_t{&triangle, index, origin + t * direction, t, triangle.Normal().Dot(direction)});
         } else {
            oppdirection.push_back(IntersectedTriangle_t{&triangle, index, origin + t * direction, -t, triangle.Normal().Dot(direction)});
         }
      }
      
   }
   std::sort(std::begin(indirection), std::end(indirection));
   std::sort(std::begin(oppdirection), std::end(oppdirection));
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function to determine which of the two input triangles are closer
/// \return Bool_t if canidate is closer than current

Bool_t TGeoTriangleMesh::IsCloserTriangle(const ClosestTriangle_t &candidate, const ClosestTriangle_t &current,
                                       const TVector3 &point) const
{
   if (std::abs(candidate.fDistance - current.fDistance) <= 0.0000005) {

      const TVector3 candidateNormal = (candidate.fClosestPoint - point).Unit();
      const TVector3 currentNormal = (current.fClosestPoint - point).Unit();
      const Double_t candidateDot = std::abs(candidateNormal.Dot(candidate.fTriangle->Normal()));
      const Double_t currentDot = std::abs(currentNormal.Dot(current.fTriangle->Normal()));

      return candidateDot > currentDot;
   }
   return candidate.fDistance < current.fDistance;
}

////////////////////////////////////////////////////////////////////////////////
/// Find the closest triangle in mesh to point
///
/// \param[in] point
/// \param[in] usedTriangleIndices
/// \return ClosestTriangle_t, contains triangle ptr, closest point on that
/// triangle to point and the distance

TGeoTriangleMesh::ClosestTriangle_t
TGeoTriangleMesh::FindClosestTriangleInMesh(const TVector3 &point, const std::vector<UInt_t> &usedTriangleIndices) const
{
   UInt_t currentindex = 0;
   auto closesTGeoTriangle = ClosestTriangle_t{};
   for (UInt_t cindex : usedTriangleIndices) {
      ClosestTriangle_t candidateCloseTGeoTriangle;
      candidateCloseTGeoTriangle.fTriangle = &fTriangles[cindex];
      candidateCloseTGeoTriangle.fClosestPoint = candidateCloseTGeoTriangle.fTriangle->ClosestPointToPoint(point);
      candidateCloseTGeoTriangle.fDistance = (candidateCloseTGeoTriangle.fClosestPoint-point).Mag();
      candidateCloseTGeoTriangle.fIndex = static_cast<Int_t>(cindex);

      if (IsCloserTriangle(candidateCloseTGeoTriangle, closesTGeoTriangle, point)) {
         closesTGeoTriangle = candidateCloseTGeoTriangle;
      }

      ++currentindex;
   }

   return closesTGeoTriangle;
}

////////////////////////////////////////////////////////////////////////////////
/// Find min and max corners
///
/// \param[out] min
/// \param[out] max

void TGeoTriangleMesh::ExtremaOfMeshHull(TVector3 &min, TVector3 &max) const
{
   for (const TVector3 &point : fPoints) {
      for (Int_t component = 0; component < 3; ++component) {
         if (point[component] < min[component]) {
            min[component] = point[component];
         }
         if (max[component] < point[component]) {
            max[component] = point[component];
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return a vector of allowed indices for this mesh (just the series of
/// (0,1,....,fTriangles.size()-1))
///
/// \param[out] indices

void TGeoTriangleMesh::TriangleMeshIndices(std::vector<UInt_t> &indices) const
{
   for (const TGeoTriangle &triangle : fTriangles) {
      const auto &triangleIndices = triangle.Indices();
      std::copy(triangleIndices.begin(), triangleIndices.end(), std::back_inserter(indices));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Resize the shape by scaling vertices within maxsize and center to origin

void TGeoTriangleMesh::ResizeCenter(Double_t maxsize)
{
   TVector3 origin, halflengths;
   const Double_t min_val = -std::numeric_limits<double>::max();
   const Double_t max_val = std::numeric_limits<double>::max();
   TVector3 min{max_val, max_val, max_val};
   TVector3 max{min_val, min_val, min_val};
   ExtremaOfMeshHull(min, max);
   origin = (min + max) * 0.5;
   halflengths = max - origin;
   const Double_t maxedge = TMath::Max(TMath::Max(halflengths.X(), halflengths.Y()), halflengths.Z());
   const Double_t scale = maxsize / maxedge;
   for (size_t i = 0; i < fPoints.size(); ++i) {
      fPoints[i] = scale * (fPoints[i] - origin);
   }
   //After the points are changed, the triangles need to recompute center and normals
   for (auto &tri : fTriangles) {
      tri.Setup();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check closure of the solid and check/fix flipped normals

Bool_t TGeoTriangleMesh::CheckClosure(Bool_t fixFlipped, Bool_t verbose)
{
   const size_t numberOfTriangles{fTriangles.size()};
   std::vector<Int_t> nn{};
   nn.resize(numberOfTriangles);
   std::vector<Bool_t> flipped{};
   flipped.resize(numberOfTriangles);
   Bool_t hasorphans = kFALSE;
   Bool_t hasflipped = kFALSE;
   for (size_t i = 0; i < numberOfTriangles; ++i) {
      nn[i] = 0;
      flipped[i] = kFALSE;
   }

   for (size_t icrt = 0; icrt < numberOfTriangles; ++icrt) {
      // all neighbours checked?
      if (nn[icrt] >= TGeoTriangle::sNumberOfVertices)
         continue;
      for (size_t i = icrt + 1; i < numberOfTriangles; ++i) {
         Bool_t needsflip = kFALSE;
         Bool_t isneighbour = fTriangles[icrt].IsNeighbour(fTriangles[i], needsflip);
         flipped[i] = needsflip;
         if (isneighbour) {
            if (flipped[icrt])
               flipped[i] = !flipped[i];
            if (flipped[i])
               hasflipped = kTRUE;
            nn[icrt]++;
            nn[i]++;
            if (nn[icrt] == TGeoTriangle::sNumberOfVertices)
               break;
         }
      }
      if (nn[icrt] < TGeoTriangle::sNumberOfVertices)
         hasorphans = kTRUE;
   }

   if (hasorphans && verbose) {
      Error("Check", "Tessellated solid %s has following not fully connected facets:", GetName());
      for (size_t icrt = 0; icrt < numberOfTriangles; ++icrt) {
         if (nn[icrt] < TGeoTriangle::sNumberOfVertices) {
            std::cout << icrt << " (" << TGeoTriangle::sNumberOfVertices << " edges, " << nn[icrt] << " neighbours)\n";
         }
      }
   }
   Int_t nfixed = 0;
   if (hasflipped) {
      if (verbose)
         Warning("Check", "Tessellated solid %s has following facets with flipped normals:", GetName());
      for (size_t icrt = 0; icrt < numberOfTriangles; ++icrt) {
         if (flipped[icrt]) {
            if (verbose)
               std::cout << icrt << "\n";
            if (fixFlipped) {
               fTriangles[icrt].Flip();
               nfixed++;
            }
         }
      }
      if (nfixed && verbose)
         Info("Check", "Automatically flipped %d facets to match first defined facet", nfixed);
   }
   Bool_t brokenTriangle = kFALSE;
   for (size_t icrt = 0; icrt < numberOfTriangles; ++icrt) {
      if (!fTriangles[icrt].IsValid()) {
         std::cerr << "Triangle " << icrt << " is not valid\n";
         brokenTriangle = kTRUE;
      }
   }
   hasorphans |= brokenTriangle;
   return !hasorphans;
}

////////////////////////////////////////////////////////////////////////////////
/// Pass the pointer to the point vector of the mesh to the triangles
/// after streaming from a file

void TGeoTriangleMesh::SetupTriangles()
{
   for (auto &triangle : fTriangles) {
      triangle.SetPoints(&fPoints);
   }
}

////////////////////////////////////////////////////////////////////////////////
void TGeoTriangleMesh::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      TGeoTriangleMesh::Class()->ReadBuffer(R__b, this);
      SetupTriangles();
   } else {
      TGeoTriangleMesh::Class()->WriteBuffer(R__b, this);
   }
}

}; // namespace Tessellated