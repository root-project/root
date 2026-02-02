// @(#)root/geom:$Id$// Author: Andrei Gheata   24/10/01

// Contains() and DistFromOutside/Out() implemented by Mihaela Gheata
// navigation functionality implemented by Ben Salisbury
// DOI: 10.1051/epjconf/202533701022
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoTessellated
\ingroup Geometry_classes

Tessellated solid class. It is composed by a set of planar faces having triangular
shape. Uses ray-casting methods to compute navigation functionality (analog to
G4TessellatedSolid).
As the processing for the navigation functionality scales with the number of
triangle faces, a partitioning structure such as the Tessellated::TOctree or the
Tessellated::TBVH structures may be used to speed up processing. The benefits are not
guaranteed. Indeed, they may come with significant processing overhead for simple (low
triangle count) shapes.


*/

#include "TGeoTessellated.h"

#include <cmath>    // for sqrt, abs
#include <iostream> // for operator<<, basic_ostream, basic_ostream...
#include <limits>   // for numeric_limits
#include <numeric>  // for iota, accumulate
#include <utility>  // for move

#include "TBuffer.h"                  // for TBuffer
#include "TBuffer3D.h"                // for TBuffer3D, TBuffer3D::kRaw, TBuffer3D::k...
#include "TBuffer3DTypes.h"           // for TBuffer3DTypes, TBuffer3DTypes::kGeneric
#include "TClass.h"                   // for TClass
#include "TGeoShape.h"                // for TGeoShape
#include "TString.h"                  // for operator<<
#include "Tessellated/TGeoTriangle.h" // for TGeoTriangle
#include "Math/Vector3D.h"            // for ROOT::Math::XYZVector, operator*, operator+

class TGeoMatrix;
///////////////////////////////////////////////////////////////////////////////
ClassImp(TGeoTessellated);

////////////////////////////////////////////////////////////////////////////////
/// default constructor

TGeoTessellated::TGeoTessellated() : TGeoBBox(), fMesh(nullptr), fPartitioningStruct(nullptr)
{
   fTimer.Stop();
   fTimer.Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// default constructor

TGeoTessellated::TGeoTessellated(const char *name)
   : TGeoBBox(name, 0, 0, 0), fMesh(nullptr), fPartitioningStruct(nullptr)
{
   fTimer.Stop();
   fTimer.Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TGeoTessellated::~TGeoTessellated()
{
   if (fPrintTime) {
      std::cout << "TGeoTessellated " << fMesh->GetMeshFile() << " took Real time " << fTimer.RealTime()
                << " s, CPU time " << fTimer.CpuTime() << "s" << std::endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set the mesh by passing ownership of mesh to TGeoTessellated

void TGeoTessellated::SetMesh(std::unique_ptr<TGeoTriangleMesh> mesh)
{
   // fTimer.Start(kFALSE);
   fMesh = std::move(mesh);
   fUsedTriangles.resize(fMesh->Triangles().size());
   std::iota(fUsedTriangles.begin(), fUsedTriangles.end(), 0);
   ComputeBBox();

   // fTimer.Stop();
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Does the geometry contain the point represented with
/// \param[in] pointa
/// \return Bool_t

bool TGeoTessellated::Contains(const Double_t *pointa) const
{
   fTimer.Start(kFALSE);
   if (!TGeoBBox::Contains(pointa)) {
      fTimer.Stop();
      return false;
   }

   ROOT::Math::XYZVector point{};
   point.SetCoordinates(pointa);

   if (fPartitioningStruct != nullptr) {
      bool result = fPartitioningStruct->IsPointContained(point);
      fTimer.Stop();
      return result;
   }

   ROOT::Math::XYZVector dir{0.0, 0.0, 1.0};
   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> indir{};
   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> oppdir{};
   fMesh->FindClosestIntersectedTriangles(point, dir, fUsedTriangles, indir, oppdir);

   if (indir.empty() || oppdir.empty()) {
      if ((!indir.empty() && indir[0].fDistance < TGeoShape::Tolerance() * 10) ||
          (!oppdir.empty() && oppdir[0].fDistance < TGeoShape::Tolerance() * 10)) {
         fTimer.Stop();
         return true;
      }
      fTimer.Stop();
      return false;
   }

   if (indir[0].fDirDotNormal > 0 && oppdir[0].fDirDotNormal < 0) {
      fTimer.Stop();
      return true;
   }

   fTimer.Stop();
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Distance from point pointa to geometry surface in direction dira from the inside
///
/// \param[in] pointa
/// \param[in] dira
/// \param[in] iact
/// \param[in] step
/// \param[out] safe
/// \return Double_t
///

Double_t TGeoTessellated::DistFromInside(const Double_t *pointa, const Double_t *dira, Int_t iact, Double_t step,
                                         Double_t *safe) const
{
   fTimer.Start(kFALSE);
   if (safe != nullptr) {
      *safe = TGeoShape::Big();

      if (iact < 3) {
         *safe = Safety(pointa, true);
      }
      if (iact == 0) {
         fTimer.Stop();
         return TGeoShape::Big();
      }
      if (iact == 1 && step < *safe) {
         fTimer.Stop();
         return step; // TGeoShape::Big(); // if one reads the description returning Big is wrong
      }
   }

   ROOT::Math::XYZVector point{};
   point.SetCoordinates(pointa);
   ROOT::Math::XYZVector dir{};
   dir.SetCoordinates(dira);
   dir = dir.Unit();

   if (fPartitioningStruct != nullptr) {
      auto result = fPartitioningStruct->DistanceInDirection(point, dir, true);
      fTimer.Stop();
      return result;
   }

   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> indir{};
   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> oppdir{};
   fMesh->FindClosestIntersectedTriangles(point, dir, fUsedTriangles, indir, oppdir);
   size_t size = indir.size();
   size_t counter = 0;

   while (counter < size) {
      if (indir[counter].fDirDotNormal < 0) {
         ++counter;
      } else {
         fTimer.Stop();
         return indir[counter].fDistance;
      }
   }
   if ((!oppdir.empty() && oppdir[0].fDirDotNormal > 0) || (oppdir.empty() && indir.empty())) {
      fTimer.Stop();
      return 0;
   }

   std::cerr
      << "TGeoTessellated::DistFromInside((" << pointa[0] << "," << pointa[1] << ", " << pointa[2] << "),"
      << "(" << dira[0] << "," << dira[1] << ", " << dira[2] << "),...) found " << indir.size()
      << " triangles in direction, or all triangles are parallel to direction (even though we are in the geometry)"
      << " -> We must be hitting the edge of two triangles. We reshoot from a slightly moved point" << std::endl;
   ROOT::Math::XYZVector orthogonal = Tessellated::XYZVectorHelper::Orthogonal(dir);
   Tessellated::XYZVectorHelper::SetMag(orthogonal, 1e-6);
   Double_t npointa[3] = {point.X() - orthogonal.X(), point.Y() - orthogonal.Y(), point.Z() - orthogonal.Z()};
   fTimer.Stop();
   return DistFromInside(npointa, dira, iact, step, safe);
}

////////////////////////////////////////////////////////////////////////////////
/// Distance from point pointa to geometry surface in direction dira from the outside
///
/// \param[in] pointa
/// \param[in] dira
/// \param[in] iact
/// \param[in] step
/// \param[out] safe
/// \return Double_t
///

Double_t TGeoTessellated::DistFromOutside(const Double_t *pointa, const Double_t *dira, Int_t iact, Double_t step,
                                          Double_t *safe) const
{
   fTimer.Start(kFALSE);
   if (safe != nullptr) {
      *safe = TGeoShape::Big();

      if (iact < 3) {
         *safe = Safety(pointa, false);
      }
      if (iact == 0) {
         fTimer.Stop();
         return TGeoShape::Big();
      }
      if (iact == 1 && step < *safe) {
         fTimer.Stop();
         return TGeoShape::Big(); // might be misinterpreting the description, but returning Big seems wrong, should be
                                  // step
      }
   }

   ROOT::Math::XYZVector point{};
   point.SetCoordinates(pointa);
   ROOT::Math::XYZVector dir{};
   dir.SetCoordinates(dira);
   dir = dir.Unit();

   if (fPartitioningStruct != nullptr) {
      auto result = fPartitioningStruct->DistanceInDirection(point, dir, false);
      fTimer.Stop();
      return result;
   }

   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> indir{};
   std::vector<TGeoTriangleMesh::IntersectedTriangle_t> oppdir{};
   fMesh->FindClosestIntersectedTriangles(point, dir, fUsedTriangles, indir, oppdir);
   size_t size = indir.size();
   size_t counter = 0;

   while (counter < size) {
      if (indir[counter].fDirDotNormal <= TGeoShape::Tolerance()) {
         fTimer.Stop();
         return indir[counter].fDistance - 2 * TGeoShape::Tolerance();
      } else {
         ++counter;
      }
   }

   if (size > 0 && oppdir.empty()) {
      std::cerr << "TGeoTessellated::DistFromOutside((" << pointa[0] << "," << pointa[1] << ", " << pointa[2] << "), ("
                << dira[0] << "," << dira[1] << ", " << dira[2] << "),...) found " << indir.size()
                << " triangles in direction, but all facing towards direction -> We must be hitting the edge of two "
                   "triangles. We reshoot from a slightly moved point"
                << std::endl;
      ROOT::Math::XYZVector orthogonal = Tessellated::XYZVectorHelper::Orthogonal(dir);
      Tessellated::XYZVectorHelper::SetMag(orthogonal, 1e-6);
      Double_t npointa[3] = {point.X() - orthogonal.X(), point.Y() - orthogonal.Y(), point.Z() - orthogonal.Z()};
      fTimer.Stop();
      return DistFromOutside(npointa, dira, iact, step, safe);
   }
   fTimer.Stop();

   return TGeoShape::Big();
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the shortest distance from point pointa to the surface of the geometry
///
/// \param[in] pointa
/// \param[in] inside
/// \return Double_t
///

Double_t TGeoTessellated::Safety(const Double_t *pointa, bool inside) const
{
   fTimer.Start(kFALSE);
   if (inside == false && TGeoBBox::Contains(pointa) == false) {
      Double_t result = TGeoBBox::Safety(pointa, inside);
      fTimer.Stop();
      return result;
   }
   ROOT::Math::XYZVector point{};
   point.SetCoordinates(pointa);
   if (fPartitioningStruct != nullptr) {
      Double_t result = fPartitioningStruct->GetSafetyDistance(point);
      fTimer.Stop();
      return result;
   }
   Double_t result = fMesh->FindClosestTriangleInMesh(point, fUsedTriangles).fDistance;
   fTimer.Stop();
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the normal of the surface closest to point pointa
///
/// \param[in] pointa
/// \param[in] dira
/// \param[out] norm
///
/// ComputeNormal can return different results when using partitioning structures
/// and when not, or different partitioning structures. Reason is, that for
/// points where the closest point lies on the edge of two triangles, the chosen
/// triangle can differ and with it the found normal.

void TGeoTessellated::ComputeNormal(const Double_t *pointa, const Double_t *dira, Double_t *norm) const
{

   fTimer.Start(kFALSE);
   ROOT::Math::XYZVector point{};
   point.SetCoordinates(pointa);
   ROOT::Math::XYZVector dir{};
   dir.SetCoordinates(dira);

   ROOT::Math::XYZVector trianglenormal{0, 0, 0};

   if (fPartitioningStruct != nullptr) {
      TGeoTriangleMesh::ClosestTriangle_t closesTGeoTriangle = fPartitioningStruct->GetClosestTriangle(point);
      trianglenormal = closesTGeoTriangle.fTriangle->Normal();
   } else {
      TGeoTriangleMesh::ClosestTriangle_t closesTGeoTriangle = fMesh->FindClosestTriangleInMesh(point, fUsedTriangles);
      trianglenormal = closesTGeoTriangle.fTriangle->Normal();
   }
   norm[0] = trianglenormal.X();
   norm[1] = trianglenormal.Y();
   norm[2] = trianglenormal.Z();

   Double_t ndotd = norm[0] * dira[0] + norm[1] * dira[1] + norm[2] * dira[2];
   if (ndotd < 0.0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
      norm[2] = -norm[2];
   }
   fTimer.Stop();
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the volume of the geometry
///
/// \return Double_t
///

Double_t TGeoTessellated::Capacity() const
{
   auto triangles = fMesh->Triangles();
   auto capacity =
      std::accumulate(triangles.begin(), triangles.end(), 0.0, [](Double_t acapacity, const TGeoTriangle &triangle) {
         return acapacity + (triangle.Center().Dot(triangle.Normal()) * triangle.Area());
      });

   return (std::abs(capacity) / 3.0);
}

////////////////////////////////////////////////////////////////////////////////
/// TODO: Purpose of function unclear, returning nullptr
///
/// \param shape
/// \param matrix
/// \return TGeoShape*
///

TGeoShape *TGeoTessellated::GetMakeRuntimeShape(TGeoShape * /*shape*/, TGeoMatrix * /*matrix*/) const
{
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// TODO: Purpose of function unclear, returning false
///
/// @return Bool_t
///

Bool_t TGeoTessellated::IsCylType() const
{
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// TODO: Purpose of function unclear, returning TGeoBBox::IsValidBox()
/// Return whether the bounding box is a valid box, as a mesh in general is
/// not a box
///

Bool_t TGeoTessellated::IsValidBox() const
{
   return TGeoBBox::IsValidBox();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding cylinder containing the geometry
///
/// \param[out] param 0. inner cylinder radius, 1. outer cylinder radius
///                   2. phi start, 3. phi end
///

void TGeoTessellated::GetBoundingCylinder(Double_t *param) const
{
   // fTimer.Start(kFALSE);
   param[0] = 0.;
   param[1] = TMath::Sqrt(fDX * fDX + fDY * fDY + fDZ * fDZ);
   param[2] = 0.;
   param[3] = 360.;
   // fTimer.Stop();
}

////////////////////////////////////////////////////////////////////////////////
/// Size of TGeoTessellated instance in bytes
/// \return Int_t
///

Int_t TGeoTessellated::GetByteCount() const
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// TODO: Purpose of function unclear, returning false
/// \param intl
/// \param list
/// \return Bool_t
///

Bool_t TGeoTessellated::GetPointsOnSegments(Int_t /*intl*/, Double_t * /*list*/) const
{
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Print information about object
///

void TGeoTessellated::InspectShape() const
{
   std::cout << " TGeoTessellated " << GetName() << " read from " << fMesh->GetMeshFile()
             << " ; NTriangles = " << fMesh->Triangles().size() << " and using " << fUsedTriangles.size()
             << " Triangles\n"
             << " bounding box: X(" << -fDX << "," << fDX << "), Y(" << -fDY << "," << fDY << "), Z(" << -fDZ << ","
             << fDZ << ")";

   std::cout << ", Origin is: (";
   for (auto value : fOrigin) {
      std::cout << value << " ";
   }
   std::cout << ")\n";

   if (fPartitioningStruct) {
      std::cout << "Partitioning structure exists." << std::endl;
      fPartitioningStruct->Print();
   } else {
      std::cout << "No partitioning structure exists." << std::endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Resize the shape by scaling vertices within maxsize and center to origin

void TGeoTessellated::ResizeCenter(Double_t maxsize)
{
   fMesh->ResizeCenter(maxsize);
   ComputeBBox();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding box containing the full geometry
///

void TGeoTessellated::ComputeBBox()
{
   Double_t min_val = -std::numeric_limits<double>::max();
   Double_t max_val = std::numeric_limits<double>::max();
   ROOT::Math::XYZVector min{max_val, max_val, max_val};
   ROOT::Math::XYZVector max{min_val, min_val, min_val};
   ROOT::Math::XYZVector mid;
   fMesh->ExtremaOfMeshHull(min, max);

   mid = ((min + max) * (1.0 / 2.0));
   fOrigin[0] = mid.X();
   fOrigin[1] = mid.Y();
   fOrigin[2] = mid.Z();

   fDX = (max.X() - fOrigin[0]);
   fDY = (max.Y() - fOrigin[1]);
   fDZ = (max.Z() - fOrigin[2]);
}

////////////////////////////////////////////////////////////////////////////////
/// Fill a TBuffer3D with all geometry vertices and segments
/// Not final. Creates a segment several times, but this is the easiest implementation for now
/// \param[out] b buffer to be filled
/// \param[in] reqSections
/// \param[in] localFrame

void TGeoTessellated::FillBuffer3D(TBuffer3D &b, Int_t reqSections, Bool_t localFrame) const
{
   TGeoShape::FillBuffer3D(b, reqSections, localFrame);
   if (reqSections != TBuffer3D::kNone) {
      size_t npnts = fMesh->Points().size();

      std::vector<UInt_t> indices;
      fMesh->TriangleMeshIndices(indices);

      size_t nseg = indices.size();
      size_t ntriangles = fMesh->Triangles().size();
      const UInt_t NumberOfCoordinates = 3;
      b.SetRawSizes(static_cast<UInt_t>(npnts), static_cast<UInt_t>(NumberOfCoordinates * npnts),
                    static_cast<UInt_t>(nseg), static_cast<UInt_t>(3 * nseg), static_cast<UInt_t>(ntriangles),
                    static_cast<UInt_t>(5 * ntriangles));

      if (((reqSections & TBuffer3D::kRawSizes) != 0) || ((reqSections & TBuffer3D::kRaw) != 0)) {
         FillBuffer3DWithPoints(b);

         if (!b.fLocalFrame) {
            TransformPoints(b.fPnts, b.NbPnts());
         }

         FillBuffer3DWithSegmentsAndPols(b, indices);

         b.SetSectionsValid(TBuffer3D::kRawSizes | TBuffer3D::kRaw);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill a TBuffer3D with the geometry vertices
///
/// \param[out] b buffer to be filled
///

void TGeoTessellated::FillBuffer3DWithPoints(TBuffer3D &b) const
{
   UInt_t pointcounter = 0;

   for (const auto &point : fMesh->Points()) {
      b.fPnts[pointcounter++] = point.X();
      b.fPnts[pointcounter++] = point.Y();
      b.fPnts[pointcounter++] = point.Z();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function to fill a TBuffer3D with the geometry segments
///
/// \param[out] b buffer to be filled
/// \param[in] indices
///

void TGeoTessellated::FillBuffer3DWithSegmentsAndPols(TBuffer3D &b, const std::vector<UInt_t> &indices) const
{
   auto segmentcounter = 0;
   auto polcounter = 0;
   for (UInt_t index = 0; index < indices.size(); index += 3) {
      b.fPols[polcounter++] = 0; // color
      b.fPols[polcounter++] = 3; // number of segments

      auto segindex = (index / 3);

      b.fSegs[segindex * 9] = 0;                                          // color
      b.fSegs[segindex * 9 + 1] = static_cast<Int_t>(indices[index]);     // segment point one
      b.fSegs[segindex * 9 + 2] = static_cast<Int_t>(indices[index + 1]); // segment point two

      b.fSegs[segindex * 9 + 3] = 0;                                      // color
      b.fSegs[segindex * 9 + 4] = static_cast<Int_t>(indices[index + 1]); // segment point one
      b.fSegs[segindex * 9 + 5] = static_cast<Int_t>(indices[index + 2]); // segment point two

      b.fSegs[segindex * 9 + 6] = 0;                                      // color
      b.fSegs[segindex * 9 + 7] = static_cast<Int_t>(indices[index + 2]); // segment point one
      b.fSegs[segindex * 9 + 8] = static_cast<Int_t>(indices[index]);     // segment point two

      // The pols ordere defines the openglviewer primitive orientation
      // if wrongly choosen, this can lead to seeing into a object
      segmentcounter += 3;
      b.fPols[polcounter++] = segmentcounter - 1;
      b.fPols[polcounter++] = segmentcounter - 2;
      b.fPols[polcounter++] = segmentcounter - 3;
   }
}

////////////////////////////////////////////////////////////////////////////////
///  return a static TBuffer3D instance containing the geometry vertices and segments
///
/// \param[in] reqSections
/// \param[in] localFrame
/// \return const TBuffer3D&
///

const TBuffer3D &TGeoTessellated::GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
{
   static TBuffer3D buf(TBuffer3DTypes::kGeneric);
   FillBuffer3D(buf, reqSections, localFrame);
   return buf;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a 3D Buffer containing geometry vertices and segments
///
/// \return TBuffer3D*
///

TBuffer3D *TGeoTessellated::MakeBuffer3D() const
{
   // fTimer.Start(kFALSE);
   TBuffer3D *buf = new TBuffer3D(TBuffer3DTypes::kGeneric);
   FillBuffer3D(*buf, TBuffer3D::kCore | TBuffer3D::kRawSizes | TBuffer3D::kRaw, kFALSE);
   // fTimer.Stop();
   return buf;
}

////////////////////////////////////////////////////////////////////////////////
/// Populate array of Double_t with meshpoints
///
/// \param[out] points
///

void TGeoTessellated::SetPoints(Double_t *points) const
{
   std::size_t pointcounter{0};
   for (const auto &point : fMesh->Points()) {
      points[pointcounter++] = point.X();
      points[pointcounter++] = point.Y();
      points[pointcounter++] = point.Z();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Populate array of Floath_t with meshpoints
///
/// \param[out] points
///

void TGeoTessellated::SetPoints(Float_t *points) const
{
   std::size_t pointcounter{0};
   for (const auto &point : fMesh->Points()) {
      points[pointcounter++] = point.X();
      points[pointcounter++] = point.Y();
      points[pointcounter++] = point.Z();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Provide number of vertices, number of segments and number of pols
///
/// \param[out] nvert
/// \param[out] nsegs
/// \param[out] npols
///

void TGeoTessellated::GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
{
   nvert = GetNmeshVertices();
   npols = GetTriangleMesh()->Triangles().size();
   nsegs = npols * 3;
}

////////////////////////////////////////////////////////////////////////////////
/// Stream TGeoTessellated object into buffer
///
/// \param[out] R__b buffer to be filled with object
///

void TGeoTessellated::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      TGeoTessellated::Class()->ReadBuffer(R__b, this);
      fMesh->Streamer(R__b);

      fUsedTriangles.resize(fMesh->Triangles().size());
      std::iota(fUsedTriangles.begin(), fUsedTriangles.end(), 0);

      if (fPartitioningStruct != nullptr) {
         fPartitioningStruct->TPartitioningI::SetTriangleMesh(fMesh.get());
      }
   } else {
      TGeoTessellated::Class()->WriteBuffer(R__b, this);
      TGeoTriangleMesh::Class()->WriteBuffer(R__b, fMesh.get());
   }
   ComputeBBox();
}
