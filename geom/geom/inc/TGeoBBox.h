// @(#)root/geom:$Id$
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoBBox
#define ROOT_TGeoBBox

#include "TGeoShape.h"

class TGeoBBox : public TGeoShape {
protected:
   // data members
   Double_t fDX;        // X half-length
   Double_t fDY;        // Y half-length
   Double_t fDZ;        // Z half-length
   Double_t fOrigin[3]; // box origin
                        // methods
   void FillBuffer3D(TBuffer3D &buffer, Int_t reqSections, Bool_t localFrame) const override;

   TGeoBBox(const TGeoBBox &) = delete;
   TGeoBBox &operator=(const TGeoBBox &) = delete;

public:
   // constructors
   TGeoBBox();
   TGeoBBox(Double_t dx, Double_t dy, Double_t dz, Double_t *origin = nullptr);
   TGeoBBox(const char *name, Double_t dx, Double_t dy, Double_t dz, Double_t *origin = nullptr);
   TGeoBBox(Double_t *param);
   // destructor
   ~TGeoBBox() override;
   // methods
   
   Double_t Capacity() const override;
   void ComputeBBox() override;
   void ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) override;
   void ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize) override;
   Bool_t Contains(const Double_t *point) const override;
   void Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const override;
   static Bool_t Contains(const Double_t *point, Double_t dx, Double_t dy, Double_t dz, const Double_t *origin);
   Bool_t CouldBeCrossed(const Double_t *point, const Double_t *dir) const override;
   Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
   Double_t DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                           Double_t *safe = nullptr) const override;
   void DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                         Double_t *step) const override;
   static Double_t DistFromInside(const Double_t *point, const Double_t *dir, Double_t dx, Double_t dy, Double_t dz,
                                  const Double_t *origin, Double_t stepmax = TGeoShape::Big());
   Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact = 1,
                            Double_t step = TGeoShape::Big(), Double_t *safe = nullptr) const override;
   void DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                          Double_t *step) const override;
   static Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Double_t dx, Double_t dy, Double_t dz,
                                   const Double_t *origin, Double_t stepmax = TGeoShape::Big());
   TGeoVolume *
   Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step) override;
   const char *GetAxisName(Int_t iaxis) const override;
   Double_t GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const override;
   void GetBoundingCylinder(Double_t *param) const override;
   const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const override;
   Int_t GetByteCount() const override { return 36; }
   virtual Double_t GetFacetArea(Int_t index = 0) const;
   virtual Bool_t GetPointsOnFacet(Int_t index, Int_t npoints, Double_t *array) const;
   Bool_t GetPointsOnSegments(Int_t npoints, Double_t *array) const override;
   Int_t
   GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const override;
   TGeoShape *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const override;
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override;
   Int_t GetNmeshVertices() const override { return 8; }
   virtual Double_t GetDX() const { return fDX; }
   virtual Double_t GetDY() const { return fDY; }
   virtual Double_t GetDZ() const { return fDZ; }
   virtual const Double_t *GetOrigin() const { return fOrigin; }
   void InspectShape() const override;
   Bool_t IsCylType() const override { return kFALSE; }
   Bool_t IsValidBox() const override { return ((fDX < 0) || (fDY < 0) || (fDZ < 0)) ? kFALSE : kTRUE; }
   virtual Bool_t IsNullBox() const { return ((fDX < 1.E-16) && (fDY < 1.E-16) && (fDZ < 1.E-16)) ? kTRUE : kFALSE; }
   TBuffer3D *MakeBuffer3D() const override;
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetBoxDimensions(Double_t dx, Double_t dy, Double_t dz, Double_t *origin = nullptr);
   void SetDimensions(Double_t *param) override;
   void SetBoxPoints(Double_t *points) const;
   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;
   void SetSegsAndPols(TBuffer3D &buffer) const override;
   void Sizeof3D() const override;

   ClassDefOverride(TGeoBBox, 1) // box primitive
};

#endif
