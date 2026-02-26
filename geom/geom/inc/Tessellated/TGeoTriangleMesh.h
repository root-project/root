// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TGeoTriangleMESH_HH
#define TGeoTriangleMESH_HH

#include <algorithm> // for copy, max
#include <numeric>   // for iota
#include <vector>    // for vector, vector<>::iterator

#include "Rtypes.h"        // for THashConsistencyHolder, ClassDefOverride
#include "RtypesCore.h"    // for UInt_t, Int_t, Bool_t
#include "TObject.h"       // for TObject
#include "TString.h"       // for TString
#include "TGeoTriangle.h"  // for TGeoTriangle
#include "Math/Vector3D.h" // for ROOT::Math::XYZVector

class TBuffer;
class TClass;
class TMemberInspector;

namespace Tessellated {

class TGeoTriangleMesh : public TObject {
public:
   enum class LengthUnit : Int_t {
      kMilliMeter,
      kCentiMeter,
      kMeter
   };

   struct IntersectedTriangle_t {
      const TGeoTriangle *fTriangle{nullptr};
      Int_t fIndex{-1};
      ROOT::Math::XYZVector fIntersectionPoint{0, 0, 0};
      Double_t fDistance{TGeoTriangle::sINF};
      Double_t fDirDotNormal{0};
      Bool_t operator<(const IntersectedTriangle_t &rhs) const
      {
         if (std::abs(fDistance - rhs.fDistance) < TGeoShape::Tolerance()) {
            return std::abs(fDirDotNormal) > std::abs(rhs.fDirDotNormal);
         }
         return fDistance < rhs.fDistance;
      }
   };

   struct ClosestTriangle_t {
      const TGeoTriangle *fTriangle{nullptr};
      ROOT::Math::XYZVector fClosestPoint{0, 0, 0};
      Double_t fDistance{TGeoTriangle::sINF};
      Int_t fIndex{-1};
   };

private:
   std::vector<ROOT::Math::XYZVector> fPoints{}; ///< vector of mesh vertices/points
   std::vector<TGeoTriangle> fTriangles{};       ///< vector of triangles forming mesh
   TString fMeshFile{""};                        ///< name of stl file read from

private:
   Bool_t IsCloserTriangle(const ClosestTriangle_t &candidate, const ClosestTriangle_t &current,
                           const ROOT::Math::XYZVector &point) const;

public:
   TGeoTriangleMesh();
   explicit TGeoTriangleMesh(const TString &meshfile);
   ~TGeoTriangleMesh() override;

   void SetTriangles(const std::vector<TGeoTriangle> &t_triangles) { fTriangles = t_triangles; }
   void SetPoints(const std::vector<ROOT::Math::XYZVector> &t_points) { fPoints = t_points; }
   const TGeoTriangle &TriangleAt(size_t index) const { return fTriangles[index]; }
   TGeoTriangle TriangleAt(size_t index) { return fTriangles[index]; }
   const std::vector<TGeoTriangle> &Triangles() const { return fTriangles; }
   const std::vector<ROOT::Math::XYZVector> &Points() const { return fPoints; }
   const ROOT::Math::XYZVector &Point(size_t index) const { return fPoints.at(index); }
   ROOT::Math::XYZVector Point(size_t index) { return fPoints.at(index); }
   std::vector<UInt_t> GetTriangleIndices() const;
   const TString &GetMeshFile() const { return fMeshFile; }
   size_t GetNumberOfTriangles() const { return fTriangles.size(); }
   size_t GetNumberOfVertices() const { return fPoints.size(); }

   void ExtremaOfMeshHull(ROOT::Math::XYZVector &min, ROOT::Math::XYZVector &max) const;
   void FindClosestIntersectedTriangles(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction,
                                        const std::vector<UInt_t> &usedTriangleIndices,
                                        std::vector<IntersectedTriangle_t> &indirection,
                                        std::vector<IntersectedTriangle_t> &againstdirection) const;

   ClosestTriangle_t
   FindClosestTriangleInMesh(const ROOT::Math::XYZVector &point, const std::vector<UInt_t> &usedTriangleIndices) const;
   void TriangleMeshIndices(std::vector<UInt_t> &indices) const;

   void ResizeCenter(Double_t maxsize);
   Bool_t CheckClosure(bool fixFlipped, bool verbose);
   void SetupTriangles();

   ClassDefOverride(TGeoTriangleMesh, 1)
};

}; // namespace Tessellated

#endif /*TGeoTriangleMESH_HH*/
