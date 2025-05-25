// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoSphere
#define ROOT_TGeoSphere

#include "TGeoBBox.h"

class TGeoSphere : public TGeoBBox {
protected:
   Int_t fNz;        // number of z planes for drawing
   Int_t fNseg;      // number of segments for drawing
   Double_t fRmin;   // inner radius
   Double_t fRmax;   // outer radius
   Double_t fTheta1; // lower theta limit
   Double_t fTheta2; // higher theta limit
   Double_t fPhi1;   // lower phi limit
   Double_t fPhi2;   // higher phi limit

   TGeoSphere(const TGeoSphere &) = delete;
   TGeoSphere &operator=(const TGeoSphere &) = delete;

public:
   // constructors
   TGeoSphere();
   TGeoSphere(Double_t rmin, Double_t rmax, Double_t theta1 = 0, Double_t theta2 = 180, Double_t phi1 = 0,
              Double_t phi2 = 360);
   TGeoSphere(const char *name, Double_t rmin, Double_t rmax, Double_t theta1 = 0, Double_t theta2 = 180,
              Double_t phi1 = 0, Double_t phi2 = 360);
   TGeoSphere(Double_t *param, Int_t nparam = 6);
   // destructor
   ~TGeoSphere() override;
   // methods
   Double_t Capacity() const override;
   void ComputeBBox() override;
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
   Double_t DistToSphere(const Double_t *point, const Double_t *dir, Double_t rsph, Bool_t check = kTRUE,
                         Bool_t firstcross = kTRUE) const;
   TGeoVolume *
   Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step) override;
   const char *GetAxisName(Int_t iaxis) const override;
   Double_t GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const override;
   void GetBoundingCylinder(Double_t *param) const override;
   const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const override;
   Int_t GetByteCount() const override { return 42; }
   TGeoShape *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const override { return nullptr; }
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override;
   Int_t GetNmeshVertices() const override;
   Int_t GetNumberOfDivisions() const { return fNseg; }
   Bool_t GetPointsOnSegments(Int_t /*npoints*/, Double_t * /*array*/) const override { return kFALSE; }
   Int_t GetNz() const { return fNz; }
   virtual Double_t GetRmin() const { return fRmin; }
   virtual Double_t GetRmax() const { return fRmax; }
   Double_t GetTheta1() const { return fTheta1; }
   Double_t GetTheta2() const { return fTheta2; }
   Double_t GetPhi1() const { return fPhi1; }
   Double_t GetPhi2() const { return fPhi2; }
   void InspectShape() const override;
   Bool_t IsCylType() const override { return kFALSE; }
   Int_t IsOnBoundary(const Double_t *point) const;
   Bool_t
   IsPointInside(const Double_t *point, Bool_t checkR = kTRUE, Bool_t checkTh = kTRUE, Bool_t checkPh = kTRUE) const;
   TBuffer3D *MakeBuffer3D() const override;
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetSphDimensions(Double_t rmin, Double_t rmax, Double_t theta1, Double_t theta2, Double_t phi1, Double_t phi2);
   virtual void SetNumberOfDivisions(Int_t p);
   void SetDimensions(Double_t *param) override;
   void SetDimensions(Double_t *param, Int_t nparam);
   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;
   void SetSegsAndPols(TBuffer3D &buff) const override;
   void Sizeof3D() const override;

   ClassDefOverride(TGeoSphere, 1) // sphere class
};

#endif
