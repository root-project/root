// @(#) :$Id$
// Author: Mihaela Gheata   03/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoHalfSpace
#define ROOT_TGeoHalfSpace

#include "TGeoBBox.h"

class TGeoHalfSpace : public TGeoBBox {
private:
   Double_t fP[3];
   Double_t fN[3];

   TGeoHalfSpace(const TGeoHalfSpace &) = delete;
   TGeoHalfSpace &operator=(const TGeoHalfSpace &) = delete;

public:
   // constructors
   TGeoHalfSpace();
   TGeoHalfSpace(const char *name, Double_t *p, Double_t *n);
   TGeoHalfSpace(Double_t *params);
   // destructor
   ~TGeoHalfSpace() override;
   // methods
   Double_t Capacity() const override { return 0.; }
   void ComputeBBox() override {}
   void ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) const override;
   void ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize) override;
   Bool_t Contains(const Double_t *point) const override;
   void Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const override;
   Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
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
   virtual Double_t *GetPoint() { return fP; }
   virtual Double_t *GetNorm() { return fN; }
   void GetBoundingCylinder(Double_t * /*param*/) const override {}
   TGeoShape *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const override { return nullptr; }
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override;
   Int_t GetNmeshVertices() const override { return 0; }
   void InspectShape() const override;
   Bool_t IsCylType() const override { return kFALSE; }
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetDimensions(Double_t *param) override;
   void SetPoints(Double_t * /*points*/) const override {}
   void SetPoints(Float_t * /*points*/) const override {}
   void Sizeof3D() const override {}

   ClassDefOverride(TGeoHalfSpace, 1) // half-space class
};

#endif
