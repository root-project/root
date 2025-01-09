// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoPgon
#define ROOT_TGeoPgon

#include "TGeoPcon.h"

#include <mutex>
#include <vector>

class TGeoPgon : public TGeoPcon {
public:
   struct ThreadData_t {
      Int_t *fIntBuffer;    //![fNedges+4] temporary int buffer array
      Double_t *fDblBuffer; //![fNedges+4] temporary double buffer array

      ThreadData_t();
      ~ThreadData_t();
   };
   ThreadData_t &GetThreadData() const;
   void ClearThreadData() const override;
   void CreateThreadData(Int_t nthreads) override;

protected:
   // data members
   Int_t fNedges;                                   // number of edges (at least one)
   mutable std::vector<ThreadData_t *> fThreadData; //! Navigation data per thread
   mutable Int_t fThreadSize;                       //! Size for the navigation data array
   mutable std::mutex fMutex;                       //! Mutex for thread data

   // internal utility methods
   Int_t GetPhiCrossList(const Double_t *point, const Double_t *dir, Int_t istart, Double_t *sphi, Int_t *iphi,
                         Double_t stepmax = TGeoShape::Big()) const;
   Bool_t IsCrossingSlice(const Double_t *point, const Double_t *dir, Int_t iphi, Double_t sstart, Int_t &ipl,
                          Double_t &snext, Double_t stepmax) const;
   void LocatePhi(const Double_t *point, Int_t &ipsec) const;
   Double_t Rpg(Double_t z, Int_t ipl, Bool_t inner, Double_t &a, Double_t &b) const;
   Double_t Rproj(Double_t z, const Double_t *point, const Double_t *dir, Double_t cphi, Double_t sphi, Double_t &a,
                  Double_t &b) const;
   Bool_t SliceCrossing(const Double_t *point, const Double_t *dir, Int_t nphi, Int_t *iphi, Double_t *sphi,
                        Double_t &snext, Double_t stepmax) const;
   Bool_t SliceCrossingIn(const Double_t *point, const Double_t *dir, Int_t ipl, Int_t nphi, Int_t *iphi,
                          Double_t *sphi, Double_t &snext, Double_t stepmax) const;
   Bool_t SliceCrossingZ(const Double_t *point, const Double_t *dir, Int_t nphi, Int_t *iphi, Double_t *sphi,
                         Double_t &snext, Double_t stepmax) const;
   Bool_t SliceCrossingInZ(const Double_t *point, const Double_t *dir, Int_t nphi, Int_t *iphi, Double_t *sphi,
                           Double_t &snext, Double_t stepmax) const;
   void SetSegsAndPolsNoInside(TBuffer3D &buff) const;

   TGeoPgon(const TGeoPgon &) = delete;
   TGeoPgon &operator=(const TGeoPgon &) = delete;

public:
   // constructors
   TGeoPgon();
   TGeoPgon(Double_t phi, Double_t dphi, Int_t nedges, Int_t nz);
   TGeoPgon(const char *name, Double_t phi, Double_t dphi, Int_t nedges, Int_t nz);
   TGeoPgon(Double_t *params);
   // destructor
   ~TGeoPgon() override;
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
   Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
   TGeoVolume *
   Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step) override;
   void GetBoundingCylinder(Double_t *param) const override;
   const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const override;
   Int_t GetByteCount() const override { return 64 + 12 * fNz; }
   TGeoShape *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const override { return nullptr; }
   void GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const override;
   Int_t GetNedges() const { return fNedges; }
   Int_t GetNmeshVertices() const override;
   Int_t GetNsegments() const override { return fNedges; }
   Bool_t GetPointsOnSegments(Int_t npoints, Double_t *array) const override
   {
      return TGeoBBox::GetPointsOnSegments(npoints, array);
   }
   void InspectShape() const override;
   TBuffer3D *MakeBuffer3D() const override;
   Double_t Safety(const Double_t *point, Bool_t in = kTRUE) const override;
   void Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const override;
   Double_t SafetyToSegment(const Double_t *point, Int_t ipl, Int_t iphi, Bool_t in, Double_t safphi,
                            Double_t safmin = TGeoShape::Big()) const;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetDimensions(Double_t *param) override;
   void SetNedges(Int_t ne)
   {
      if (ne > 2)
         fNedges = ne;
   }
   void SetPoints(Double_t *points) const override;
   void SetPoints(Float_t *points) const override;
   void SetSegsAndPols(TBuffer3D &buff) const override;
   void Sizeof3D() const override;

   ClassDefOverride(TGeoPgon, 1) // polygone class
};

#endif
