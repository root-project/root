// @(#)root/geom:$Id$
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoArb8
#define ROOT_TGeoArb8

#include "TGeoBBox.h"

#include <cstdlib>

class TGeoArb8 : public TGeoBBox {
protected:
   enum EGeoArb8Type {
      //      kArb8Trd1 = BIT(25), // trd1 type
      //      kArb8Trd2 = BIT(26), // trd2 type
      kArb8Trap = BIT(27), // planar surface trapezoid
      kArb8Tra = BIT(28)   // general twisted trapezoid
   };
   // data members
   Double_t fDz{0};           // half length in Z
   Double_t *fTwist{nullptr}; //! [4] tangents of twist angles
   Double_t fXY[8][2];        // list of vertices

   TGeoArb8(const TGeoArb8 &) = delete;
   TGeoArb8 &operator=(const TGeoArb8 &) = delete;

   void CopyTwist(Double_t *twist = nullptr);

public:
   // constructors
   TGeoArb8();
   TGeoArb8(Double_t dz, Double_t *vertices = nullptr);
   TGeoArb8(const char *name, Double_t dz, Double_t *vertices = nullptr);
   // destructor
   ~TGeoArb8() override;
   // methods
   Double_t Capacity() const override;
   void ComputeBBox() override;
   void ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) const override;
   void ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize) override;
   void ComputeTwist();
   Bool_t Contains(const Double_t *point) const override;
   void Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const override;
   Double_t DistToPlane(const Double_t *point, const Double_t *dir, Int_t ipl, Bool_t in) const;
   Double_t DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                           Double_t *safe = nullptr) const override;
   void DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                         Double_t *step) const override;
   Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact = 1,
                            Double_t step = TGeoShape::Big(), Double_t *safe = nullptr) const override;
   void DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                          Double_t *step) const override;
   TGeoVolume *
   Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step) override;
   Double_t GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const override;
   void GetBoundingCylinder(Double_t *param) const override;
   Int_t GetByteCount() const override { return 100; }
   Double_t GetClosestEdge(const Double_t *point, Double_t *vert, Int_t &isegment) const;
   Bool_t GetPointsOnFacet(Int_t /*index*/, Int_t /*npoints*/, Double_t * /*array*/) const override;
   Double_t GetDz() const { return fDz; }
   Int_t
   GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const override;
   TGeoShape *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const override { return nullptr; }
   static void GetPlaneNormal(Double_t *p1, Double_t *p2, Double_t *p3, Double_t *norm);
   Double_t *GetVertices() { return &fXY[0][0]; }
   Double_t GetTwist(Int_t iseg) const;
   Bool_t IsCylType() const override { return kFALSE; }
   static Bool_t IsSamePoint(const Double_t *p1, const Double_t *p2)
   {
      return (std::abs(p1[0] - p2[0]) < 1.E-16 && std::abs(p1[1] - p2[1]) < 1.E-16) ? kTRUE : kFALSE;
   }
   static Bool_t InsidePolygon(Double_t x, Double_t y, Double_t *pts);
   void InspectShape() const override;
   Bool_t IsTwisted() const { return !fTwist ? kFALSE : kTRUE; }
   Double_t SafetyToFace(const Double_t *point, Int_t iseg, Bool_t in) const;
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetPlaneVertices(Double_t zpl, Double_t *vertices) const;
   virtual void SetVertex(Int_t vnum, Double_t x, Double_t y);
   void SetDimensions(Double_t *param) override;
   void SetDz(Double_t dz) { fDz = dz; }
   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;
   void Sizeof3D() const override;

   ClassDefOverride(TGeoArb8, 1) // arbitrary trapezoid with 8 vertices
};

class TGeoTrap : public TGeoArb8 {
protected:
   // data members
   Double_t fTheta;  // theta angle
   Double_t fPhi;    // phi angle
   Double_t fH1;     // half length in y at low z
   Double_t fBl1;    // half length in x at low z and y low edge
   Double_t fTl1;    // half length in x at low z and y high edge
   Double_t fAlpha1; // angle between centers of x edges an y axis at low z
   Double_t fH2;     // half length in y at high z
   Double_t fBl2;    // half length in x at high z and y low edge
   Double_t fTl2;    // half length in x at high z and y high edge
   Double_t fAlpha2; // angle between centers of x edges an y axis at low z

public:
   // constructors
   TGeoTrap();
   TGeoTrap(Double_t dz, Double_t theta, Double_t phi);
   TGeoTrap(Double_t dz, Double_t theta, Double_t phi, Double_t h1, Double_t bl1, Double_t tl1, Double_t alpha1,
            Double_t h2, Double_t bl2, Double_t tl2, Double_t alpha2);
   TGeoTrap(const char *name, Double_t dz, Double_t theta, Double_t phi, Double_t h1, Double_t bl1, Double_t tl1,
            Double_t alpha1, Double_t h2, Double_t bl2, Double_t tl2, Double_t alpha2);
   // destructor
   ~TGeoTrap() override;
   Double_t DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                           Double_t *safe = nullptr) const override;
   void DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                         Double_t *step) const override;
   Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact = 1,
                            Double_t step = TGeoShape::Big(), Double_t *safe = nullptr) const override;
   void DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                          Double_t *step) const override;
   TGeoVolume *
   Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step) override;
   Double_t GetTheta() const { return fTheta; }
   Double_t GetPhi() const { return fPhi; }
   Double_t GetH1() const { return fH1; }
   Double_t GetBl1() const { return fBl1; }
   Double_t GetTl1() const { return fTl1; }
   Double_t GetAlpha1() const { return fAlpha1; }
   Double_t GetH2() const { return fH2; }
   Double_t GetBl2() const { return fBl2; }
   Double_t GetTl2() const { return fTl2; }
   Double_t GetAlpha2() const { return fAlpha2; }
   TGeoShape *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const override;
   void SetDimensions(Double_t *param) override;
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGeoTrap, 1) // G3 TRAP shape
};

class TGeoGtra : public TGeoTrap {
protected:
   // data members
   Double_t fTwistAngle; // twist angle in degrees
public:
   // constructors
   TGeoGtra();
   TGeoGtra(Double_t dz, Double_t theta, Double_t phi, Double_t twist, Double_t h1, Double_t bl1, Double_t tl1,
            Double_t alpha1, Double_t h2, Double_t bl2, Double_t tl2, Double_t alpha2);
   TGeoGtra(const char *name, Double_t dz, Double_t theta, Double_t phi, Double_t twist, Double_t h1, Double_t bl1,
            Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2, Double_t tl2, Double_t alpha2);
   // destructor
   ~TGeoGtra() override;
   Double_t DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                           Double_t *safe = nullptr) const override;
   void DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                         Double_t *step) const override;
   Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact = 1,
                            Double_t step = TGeoShape::Big(), Double_t *safe = nullptr) const override;
   void DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                          Double_t *step) const override;
   TGeoShape *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const override;
   Double_t GetTwistAngle() const { return fTwistAngle; }
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   void SetDimensions(Double_t *param) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGeoGtra, 1) // G3 GTRA shape
};

#endif
