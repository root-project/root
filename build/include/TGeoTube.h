// @(#)root/base:$Id$
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTube
#define ROOT_TGeoTube

#include "TGeoBBox.h"

class TGeoTube : public TGeoBBox {
protected:
   // data members
   Double_t fRmin; // inner radius
   Double_t fRmax; // outer radius
   Double_t fDz;   // half length
                   // methods

   TGeoTube(const TGeoTube &) = delete;
   TGeoTube &operator=(const TGeoTube &) = delete;

public:
   // constructors
   TGeoTube();
   TGeoTube(Double_t rmin, Double_t rmax, Double_t dz);
   TGeoTube(const char *name, Double_t rmin, Double_t rmax, Double_t dz);
   TGeoTube(Double_t *params);
   // destructor
   ~TGeoTube() override;
   // methods

   Double_t Capacity() const override;
   static Double_t Capacity(Double_t rmin, Double_t rmax, Double_t dz);
   void ComputeBBox() override;
   void ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) const override;
   void ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize) override;
   static void ComputeNormalS(const Double_t *point, const Double_t *dir, Double_t *norm, Double_t rmin, Double_t rmax,
                              Double_t dz);
   Bool_t Contains(const Double_t *point) const override;
   void Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const override;
   static Double_t
   DistFromInsideS(const Double_t *point, const Double_t *dir, Double_t rmin, Double_t rmax, Double_t dz);
   Double_t DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                           Double_t *safe = nullptr) const override;
   void DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                         Double_t *step) const override;
   static Double_t
   DistFromOutsideS(const Double_t *point, const Double_t *dir, Double_t rmin, Double_t rmax, Double_t dz);
   Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact = 1,
                            Double_t step = TGeoShape::Big(), Double_t *safe = nullptr) const override;
   void DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                          Double_t *step) const override;
   static void DistToTube(Double_t rsq, Double_t nsq, Double_t rdotn, Double_t radius, Double_t &b, Double_t &delta);
   Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
   TGeoVolume *
   Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step) override;
   const char *GetAxisName(Int_t iaxis) const override;
   Double_t GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const override;
   void GetBoundingCylinder(Double_t *param) const override;
   const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const override;
   Int_t GetByteCount() const override { return 48; }
   Bool_t GetPointsOnSegments(Int_t npoints, Double_t *array) const override;
   TGeoShape *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const override;
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override;
   Int_t GetNmeshVertices() const override;
   virtual Double_t GetRmin() const { return fRmin; }
   virtual Double_t GetRmax() const { return fRmax; }
   virtual Double_t GetDz() const { return fDz; }
   Bool_t HasRmin() const { return (fRmin > 0) ? kTRUE : kFALSE; }
   void InspectShape() const override;
   Bool_t IsCylType() const override { return kTRUE; }
   TBuffer3D *MakeBuffer3D() const override;
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   static Double_t
   SafetyS(const Double_t *point, Bool_t in, Double_t rmin, Double_t rmax, Double_t dz, Int_t skipz = 0);
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetTubeDimensions(Double_t rmin, Double_t rmax, Double_t dz);
   void SetDimensions(Double_t *param) override;
   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;
   void SetSegsAndPols(TBuffer3D &buff) const override;
   void Sizeof3D() const override;

   ClassDefOverride(TGeoTube, 1) // cylindrical tube class
};

class TGeoTubeSeg : public TGeoTube {
protected:
   // data members
   Double_t fPhi1; // first phi limit
   Double_t fPhi2; // second phi limit
   // Transient trigonometric data
   Double_t fS1;   // sin(phi1)
   Double_t fC1;   // cos(phi1)
   Double_t fS2;   // sin(phi2)
   Double_t fC2;   // cos(phi2)
   Double_t fSm;   // sin(0.5*(phi1+phi2))
   Double_t fCm;   // cos(0.5*(phi1+phi2))
   Double_t fCdfi; // cos(0.5*(phi1-phi2))

   void InitTrigonometry();

public:
   // constructors
   TGeoTubeSeg();
   TGeoTubeSeg(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2);
   TGeoTubeSeg(const char *name, Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2);
   TGeoTubeSeg(Double_t *params);
   // destructor
   ~TGeoTubeSeg() override;
   // methods
   void AfterStreamer() override;
   Double_t Capacity() const override;
   static Double_t Capacity(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2);
   void ComputeBBox() override;
   void ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) const override;
   void ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize) override;
   static void ComputeNormalS(const Double_t *point, const Double_t *dir, Double_t *norm, Double_t rmin, Double_t rmax,
                              Double_t dz, Double_t c1, Double_t s1, Double_t c2, Double_t s2);
   Bool_t Contains(const Double_t *point) const override;
   void Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const override;
   Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
   static Double_t DistFromInsideS(const Double_t *point, const Double_t *dir, Double_t rmin, Double_t rmax,
                                   Double_t dz, Double_t c1, Double_t s1, Double_t c2, Double_t s2, Double_t cm,
                                   Double_t sm, Double_t cdfi);
   Double_t DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                           Double_t *safe = nullptr) const override;
   void DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                         Double_t *step) const override;
   static Double_t DistFromOutsideS(const Double_t *point, const Double_t *dir, Double_t rmin, Double_t rmax,
                                    Double_t dz, Double_t c1, Double_t s1, Double_t c2, Double_t s2, Double_t cm,
                                    Double_t sm, Double_t cdfi);
   Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact = 1,
                            Double_t step = TGeoShape::Big(), Double_t *safe = nullptr) const override;
   void DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                          Double_t *step) const override;
   TGeoVolume *
   Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step) override;
   Double_t GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const override;
   void GetBoundingCylinder(Double_t *param) const override;
   const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const override;
   Int_t GetByteCount() const override { return 56; }
   Bool_t GetPointsOnSegments(Int_t npoints, Double_t *array) const override;
   TGeoShape *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const override;
   Int_t GetNmeshVertices() const override;
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override;
   Double_t GetPhi1() const { return fPhi1; }
   Double_t GetPhi2() const { return fPhi2; }
   void InspectShape() const override;
   TBuffer3D *MakeBuffer3D() const override;
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   static Double_t SafetyS(const Double_t *point, Bool_t in, Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1,
                           Double_t phi2, Int_t skipz = 0);
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetTubsDimensions(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2);
   void SetDimensions(Double_t *param) override;
   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;
   void SetSegsAndPols(TBuffer3D &buff) const override;
   void Sizeof3D() const override;

   ClassDefOverride(TGeoTubeSeg, 2) // cylindrical tube segment class
};

class TGeoCtub : public TGeoTubeSeg {
protected:
   // data members
   Double_t fNlow[3];  // normal to lower cut plane
   Double_t fNhigh[3]; // normal to higher cut plane

public:
   // constructors
   TGeoCtub();
   TGeoCtub(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2, Double_t lx, Double_t ly,
            Double_t lz, Double_t tx, Double_t ty, Double_t tz);
   TGeoCtub(const char *name, Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2, Double_t lx,
            Double_t ly, Double_t lz, Double_t tx, Double_t ty, Double_t tz);
   TGeoCtub(Double_t *params);
   // destructor
   ~TGeoCtub() override;
   // methods
   Double_t Capacity() const override;
   void ComputeBBox() override;
   void ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) const override;
   void ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize) override;
   Bool_t Contains(const Double_t *point) const override;
   void Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const override;
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
   const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const override;
   Int_t GetByteCount() const override { return 98; }
   Bool_t GetPointsOnSegments(Int_t npoints, Double_t *array) const override;
   TGeoShape *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const override;
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override;
   Int_t GetNmeshVertices() const override;
   const Double_t *GetNlow() const { return &fNlow[0]; }
   const Double_t *GetNhigh() const { return &fNhigh[0]; }
   Double_t GetZcoord(Double_t xc, Double_t yc, Double_t zc) const;
   void InspectShape() const override;
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetCtubDimensions(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2, Double_t lx,
                          Double_t ly, Double_t lz, Double_t tx, Double_t ty, Double_t tz);
   void SetDimensions(Double_t *param) override;
   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;

   ClassDefOverride(TGeoCtub, 1) // cut tube segment class
};

#endif
