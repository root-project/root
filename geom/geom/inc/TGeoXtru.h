// @(#)root/geom:$Id$
// Author: Mihaela Gheata   24/01/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoXtru
#define ROOT_TGeoXtru

#include "TGeoBBox.h"

#include <mutex>
#include <vector>

class TGeoPolygon;

class TGeoXtru : public TGeoBBox {
public:
   struct ThreadData_t {
      Int_t fSeg;         // !current segment [0,fNvert-1]
      Int_t fIz;          // !current z plane [0,fNz-1]
      Double_t *fXc;      // ![fNvert] current X positions for polygon vertices
      Double_t *fYc;      // ![fNvert] current Y positions for polygon vertices
      TGeoPolygon *fPoly; // !polygon defining section shape

      ThreadData_t();
      ~ThreadData_t();
   };
   ThreadData_t &GetThreadData() const;
   void ClearThreadData() const override;
   void CreateThreadData(Int_t nthreads) override;

protected:
   // data members
   Int_t fNvert;       // number of vertices of the 2D polygon (at least 3)
   Int_t fNz;          // number of z planes (at least two)
   Double_t fZcurrent; // current Z position
   Double_t *fX;       //[fNvert] X positions for polygon vertices
   Double_t *fY;       //[fNvert] Y positions for polygon vertices
   Double_t *fZ;       //[fNz] array of Z planes positions
   Double_t *fScale;   //[fNz] array of scale factors (for each Z)
   Double_t *fX0;      //[fNz] array of X offsets (for each Z)
   Double_t *fY0;      //[fNz] array of Y offsets (for each Z)

   mutable std::vector<ThreadData_t *> fThreadData; //! Navigation data per thread
   mutable Int_t fThreadSize;                       //! size of thread-specific array
   mutable std::mutex fMutex;                       //! mutex for thread data

   TGeoXtru(const TGeoXtru &) = delete;
   TGeoXtru &operator=(const TGeoXtru &) = delete;

   // methods
   Double_t
   DistToPlane(const Double_t *point, const Double_t *dir, Int_t iz, Int_t ivert, Double_t stepmax, Bool_t in) const;
   void GetPlaneVertices(Int_t iz, Int_t ivert, Double_t *vert) const;
   void GetPlaneNormal(const Double_t *vert, Double_t *norm) const;
   Bool_t IsPointInsidePlane(const Double_t *point, Double_t *vert, Double_t *norm) const;
   Double_t SafetyToSector(const Double_t *point, Int_t iz, Double_t safmin, Bool_t in);
   void SetIz(Int_t iz);
   void SetSeg(Int_t iseg);

public:
   // constructors
   TGeoXtru();
   TGeoXtru(Int_t nz);
   TGeoXtru(Double_t *param);
   // destructor
   ~TGeoXtru() override;
   // methods
   Double_t Capacity() const override;
   void ComputeBBox() override;
   void ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm) const override;
   void ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize) override;
   Bool_t Contains(const Double_t *point) const override;
   void Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const override;
   Bool_t DefinePolygon(Int_t nvert, const Double_t *xv, const Double_t *yv);
   virtual void DefineSection(Int_t snum, Double_t z, Double_t x0 = 0., Double_t y0 = 0., Double_t scale = 1.);
   Double_t DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact = 1, Double_t step = TGeoShape::Big(),
                           Double_t *safe = nullptr) const override;
   void DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                         Double_t *step) const override;
   Double_t DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact = 1,
                            Double_t step = TGeoShape::Big(), Double_t *safe = nullptr) const override;
   void DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize,
                          Double_t *step) const override;
   Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
   void DrawPolygon(Option_t *option = "");
   const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const override;
   //   virtual Int_t         GetByteCount() const {return 60+12*fNz;}
   Int_t GetNz() const { return fNz; }
   Int_t GetNvert() const { return fNvert; }
   Double_t GetX(Int_t i) const { return (i < fNvert && i > -1 && fX) ? fX[i] : -1.0E10; }
   Double_t GetY(Int_t i) const { return (i < fNvert && i > -1 && fY) ? fY[i] : -1.0E10; }
   Double_t GetXOffset(Int_t i) const { return (i < fNz && i > -1 && fX0) ? fX0[i] : 0.0; }
   Double_t GetYOffset(Int_t i) const { return (i < fNz && i > -1 && fY0) ? fY0[i] : 0.0; }
   Double_t GetScale(Int_t i) const { return (i < fNz && i > -1 && fScale) ? fScale[i] : 1.0; }
   Double_t *GetZ() const { return fZ; }
   Double_t GetZ(Int_t ipl) const;
   TGeoShape *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const override { return nullptr; }
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override;
   Int_t GetNmeshVertices() const override;
   void InspectShape() const override;
   TBuffer3D *MakeBuffer3D() const override;
   Double_t &Z(Int_t ipl) { return fZ[ipl]; }
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetCurrentZ(Double_t z, Int_t iz);
   void SetCurrentVertices(Double_t x0, Double_t y0, Double_t scale);
   void SetDimensions(Double_t *param) override;
   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;
   void SetSegsAndPols(TBuffer3D &buff) const override;
   void Sizeof3D() const override;

   ClassDefOverride(TGeoXtru, 3) // extruded polygon class
};

#endif
