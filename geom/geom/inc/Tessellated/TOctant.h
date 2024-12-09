// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TOCTANT_HH
#define TOCTANT_HH

#include <vector> // for vector

#include "Rtypes.h"     // for THashConsistencyHolder, ClassDefOverride
#include "RtypesCore.h" // for UInt_t, Bool_t, Double_t, kFALSE
#include "TObject.h"    // for TObject
#include "TVector3.h"   // for TVector3

class TBuffer;
class TClass;
class TMemberInspector;

namespace Tessellated {

class TGeoTriangleMesh;
class TGeoTriangle;

struct OctantBounds_t {
   OctantBounds_t() = default;
   OctantBounds_t(const TVector3 &min, const TVector3 &max) : fMin(min), fMax(max) {}
   TVector3 fMin;
   TVector3 fMax;
};

struct OctreeConfig_t {
   OctreeConfig_t() = default;
   OctreeConfig_t(const OctantBounds_t &octantbounds, const TGeoTriangleMesh *mesh, const std::vector<UInt_t> *triangles,
                  UInt_t maxdepth = 4, UInt_t maxtriangles = 1, UInt_t currentdepth = 0)
      : fOctantBounds(octantbounds),
        fMesh(mesh),
        fContainedTriangles(triangles),
        fMaxDepth(maxdepth),
        fMaxTriangles(maxtriangles),
        fCurrentTreeDepth(currentdepth)
   {
   }
   OctantBounds_t fOctantBounds;

   const TGeoTriangleMesh *fMesh{nullptr};
   const std::vector<UInt_t> *fContainedTriangles{nullptr};

   UInt_t fMaxDepth{5};
   UInt_t fMaxTriangles{2};
   UInt_t fCurrentTreeDepth{0};
   Bool_t fAccurateSafety{kTRUE};
};

class TOctant : public TObject {
public:
   static unsigned int sNumberOfInsideOctants;
   static unsigned int sNumberOfOutsideOctants;
   static unsigned int sNumberOfLeafOctants;
   static const Double_t sAccuracy;
   enum class State : int { MIXED, INSIDE, OUTSIDE, UNKNOWN };

   static const UInt_t sNUMBER_OF_CHILDREN = 8;

private:
   TVector3 fMin;
   TVector3 fMax;

   std::vector<UInt_t> fContainedTriangles{};
   UInt_t fCurrentTreeDepth{};
   TOctant *fChildren[sNUMBER_OF_CHILDREN];
   TOctant *fParent{nullptr};
   State fState{State::UNKNOWN};

private:
   void InitializeChildren();
   void SetupOctant(const OctreeConfig_t &octantconfig);
   void SetState(const TGeoTriangleMesh *mesh);
   std::vector<OctantBounds_t> CreateChildBounds(const OctantBounds_t &bounds) const;
   std::vector<UInt_t> ContainedTriangles(const OctantBounds_t &octant, const std::vector<UInt_t> &parentstriangles,
                                          const TGeoTriangleMesh *mesh) const;

   struct ThreeVector3s_t {
      TVector3 vec1;
      TVector3 vec2;
      TVector3 vec3;
   };

   Bool_t TriangleOctantBoundsIntersection(const TGeoTriangle &triangle, const OctantBounds_t &octantbounds) const;
   Bool_t IsNormalAxisSeparating(const ThreeVector3s_t &triVertices, Int_t component, const TVector3 &extents) const;
   Bool_t IsSeparatingAxis(const ThreeVector3s_t &triVertices, const TVector3 &axis, const TVector3 &edge,
                           const TVector3 &extents) const;

public:
   TOctant();
   explicit TOctant(const OctreeConfig_t &octantconfig);
   ~TOctant() override;

   void CreateChildOctants(const OctreeConfig_t &octantconfig);
   Bool_t IsContainedByOctant(const TVector3 &point, Double_t epsilon) const;
   Double_t GetMinDistanceToBoundaries(const TVector3 &point) const;
   Double_t GetMinDistance(const TVector3 &point) const;

   inline const std::vector<UInt_t> &GetContainedTriangles() const { return fContainedTriangles; }
   inline const TOctant *GetChild(size_t index) const { return fChildren[index]; }

   Bool_t IsLeaf() const { return fChildren[0] == nullptr; }
   const TVector3 &GetLowerCorner() const { return fMin; }
   const TVector3 &GetUpperCorner() const { return fMax; }

   /// What is the state of the octant? Does it contain triangles (then it
   /// is MIXED), or is this octant inside or outside of the geometry surface
   State GetState() const { return fState; }
   UInt_t GetNumberOfOctants() const;
   void SetParent(TOctant *t_parent) { fParent = t_parent; }
   const TOctant *GetParent() const { return fParent; }

   void Print(Option_t *option = "") const override;

   ClassDefOverride(TOctant, 1)
};

}; // namespace Tessellated

#endif /*TOCTANT_HH*/
