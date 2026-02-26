// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TOCTREE_HH
#define TOCTREE_HH

#include <utility> // for pair
#include <vector>  // for vector

#include "Rtypes.h"           // for THashConsistencyHolder, ClassDefOverride
#include "RtypesCore.h"       // for Double_t, Bool_t, Int_t, kFALSE, UInt_t
#include "TOctant.h"          // for OctreeConfig_t (ptr only), OctantBounds_...
#include "TPartitioningI.h"   // for TPartitioningI
#include "TGeoTriangle.h"     // for TGeoTriangle
#include "TGeoTriangleMesh.h" // for TGeoTriangleMesh, TGeoTriangleMesh::ClosestTri...
#include "Math/Vector3D.h"    // for ROOT::Math::XYZVector

class TGeoTessellated;
class TBuffer;
class TClass;
class TMemberInspector;

namespace Tessellated {

class TOctree : public TPartitioningI {
private:
   TOctant *fRoot{nullptr};
   Bool_t fIsSetup{kFALSE};

   mutable ROOT::Math::XYZVector fDirection{};
   mutable ROOT::Math::XYZVector fOrigin{};
   Bool_t fOriginInside{kFALSE};
   ROOT::Math::XYZVector fOffset{}; ///< kept for backward compatibility
   ROOT::Math::XYZVector fScale{};

   Bool_t fAccurateSafety{kTRUE};
   UInt_t fMaxLayer{};
   UInt_t fMaxPower{};
   mutable UInt_t fIndexByte{};
   double fTolerance{1e-19}; // This value is used to determine
                             // when a direction component in the Revelles Octree Traversal
                             // is considered to be zero, i.e. when a ray direction is parallel
                             // to one of the axes.

private:
   Double_t DistFromInside(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction,
                           Bool_t isorigininside,
                           const std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections);
   Double_t DistFromOutside(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction,
                            Bool_t isorigininside,
                            const std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections);

   Bool_t CheckFacesInOctant(const TOctant *octant,
                             std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections) const;
   Bool_t FindClosestFacePoint(ROOT::Math::XYZVector origin, ROOT::Math::XYZVector direction,
                               std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections) const;
   Bool_t ProcessSubtree(Double_t tx0, Double_t ty0, Double_t tz0, Double_t tx1, Double_t ty1, Double_t tz1,
                         const TOctant *octant, const ROOT::Math::XYZVector &origin,
                         const ROOT::Math::XYZVector &direction,
                         std::vector<TGeoTriangleMesh::IntersectedTriangle_t> &triangleIntersections) const;
   Int_t FindNextNodeIndex(Double_t txM, Int_t x, Double_t tyM, Int_t y, Double_t tzM, Int_t z) const;
   Int_t FindFirstNodeIndex(Double_t tx0, Double_t ty0, Double_t tz0, Double_t txM, Double_t tyM, Double_t tzM) const;

   void GetOctants(TOctant const *tmp, std::vector<TOctant const *> &octants) const;
   TGeoTriangleMesh::ClosestTriangle_t
   GetSafetyInSphere(const ROOT::Math::XYZVector &point, const TGeoTriangleMesh::ClosestTriangle_t &candidate) const;
   void FindOctantsInSphere(Double_t radius, const ROOT::Math::XYZVector &point, TOctant const *octant,
                            std::vector<std::pair<const TOctant *, Double_t>> &octants) const;

public:
   static std::unique_ptr<TOctree>
   CreateOctree(const TGeoTessellated *tsl, UInt_t maxdepth, UInt_t maxtriangles, Bool_t accurateSafety = kTRUE);

   TOctree();
   explicit TOctree(const OctreeConfig_t &octreeconfig);
   ~TOctree() override;

   void SetTolerance(double tolerance) { fTolerance = tolerance; }
   double GetTolerance() const { return fTolerance; }
   void SetupOctree(const OctreeConfig_t &octreeconfig);
   const TOctant *GetRelevantOctant(const ROOT::Math::XYZVector &point) const;
   virtual Bool_t IsPointContained(const ROOT::Math::XYZVector &point) const override;
   virtual Double_t GetSafetyDistance(const ROOT::Math::XYZVector &point) const override;
   TGeoTriangleMesh::ClosestTriangle_t GetClosestTriangle(const ROOT::Math::XYZVector &point) const override;
   TGeoTriangleMesh::ClosestTriangle_t GetSafetyDistanceAccurate(const ROOT::Math::XYZVector &point) const;

   virtual Double_t DistanceInDirection(const ROOT::Math::XYZVector &origin, const ROOT::Math::XYZVector &direction,
                                        Bool_t isorigininside) override;

   Bool_t IsSetup() const { return fIsSetup; }
   UInt_t GetNumberOfOctants() const { return fRoot->GetNumberOfOctants(); }
   std::vector<TOctant const *> GetLeafOctants() const;

   virtual void Print(Option_t *opt = nullptr) const override;

   ClassDefOverride(TOctree, 1)
};

}; // namespace Tessellated

#endif /*TOCTREE_HH*/
