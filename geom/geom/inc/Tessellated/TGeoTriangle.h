// @(#)root/geom:$Id$// Author: Ben Salisbury   21/11/24

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TGeoTriangle_HH
#define TGeoTriangle_HH

#include <cmath>  // for abs
#include <array>  // for array
#include <vector> // for vector
#include <limits> // for numeric_limits

#include "Rtypes.h"     // for THashConsistencyHolder, ClassDefOverride
#include "RtypesCore.h" // for Double_t, Bool_t, Int_t, UInt_t
#include "TGeoShape.h"  // for TGeoShape
#include "TObject.h"    // for TObject
#include "TVector3.h"   // for TVector3

class TBuffer;
class TClass;
class TMemberInspector;

namespace Tessellated {

namespace TGeoTriangleInternal {
Bool_t EqualTo(Double_t a, Double_t b, Double_t accuracy);
Bool_t SmallerThan(Double_t a, Double_t b, Double_t accuracy);
Bool_t LargerThan(Double_t a, Double_t b, Double_t accuracy);
}; // namespace TGeoTriangleInternal

class TGeoTriangle : public TObject {
public:
   static constexpr Double_t sAccuracy = 1e-09;
   static constexpr Double_t sINF = std::numeric_limits<double>::infinity();
   static const UInt_t sNumberOfVertices = 3;

private:
   const std::vector<TVector3> *fPoints{nullptr}; ///<! non owning pointer to vector of points (owned by TGeoTriangleMesh)
   std::array<UInt_t, sNumberOfVertices> fIndices{}; ///< triangle only knows the three indices of the points forming it

   TVector3 fCenter{}; ///<! center of triangle
   TVector3 fNormal{}; ///<! normal of triangle
   Double_t fRadiusSqr{
      0}; ///<! squared distance from center to furthes corner (for quick checks if point is on triangle or not)

private:
   TVector3 CalculateCenter() const;
   TVector3 CalculateNormal() const;

   TVector3 ClosestPointOfEdgesToPoint(const TVector3 &point) const;
   TVector3 ClosestPointOfEdgeToPoint(const TVector3 &point, const TVector3 &edge, const TVector3 &edgedirection) const;

public:
   TGeoTriangle() : TObject() {} /* = default */
   explicit TGeoTriangle(const std::array<UInt_t, 3> &indices);
   TGeoTriangle(const std::vector<TVector3> *points, const std::array<UInt_t, 3> &indices);
   ~TGeoTriangle() override {} /* = default */

   void Setup();
   void Flip();
   Bool_t IsValid() const;
   Bool_t IsNeighbour(const TGeoTriangle &other, Bool_t &requireFlip) const;
   double DistanceFrom(const TVector3 &origin, const TVector3 &direction) const;
   TVector3 ClosestPointToPoint(const TVector3 &point) const;
   Double_t Area() const;
   const std::array<UInt_t, 3> &Indices() const { return fIndices; }
   UInt_t Index(size_t i) const { return fIndices.at(i); }
   const TVector3 &Point(size_t index) const { return fPoints->at(fIndices[index]); }
   void SetPoints(const std::vector<TVector3> *points)
   {
      fPoints = points;
      Setup();
   }

   const TVector3 &Center() const { return fCenter; }
   const TVector3 &Normal() const { return fNormal; }
   void Print(Option_t *option = "") const override;

   ClassDefOverride(TGeoTriangle, 1)
};

}; // namespace Tessellated

#endif /*TGeoTriangle_HH*/
