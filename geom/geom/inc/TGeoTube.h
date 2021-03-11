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

class TGeoTube : public TGeoBBox
{
protected :
// data members
   Double_t              fRmin; // inner radius
   Double_t              fRmax; // outer radius
   Double_t              fDz;   // half length
// methods

   TGeoTube(const TGeoTube&) = delete;
   TGeoTube& operator=(const TGeoTube&) = delete;

public:
   // constructors
   TGeoTube();
   TGeoTube(Double_t rmin, Double_t rmax, Double_t dz);
   TGeoTube(const char * name, Double_t rmin, Double_t rmax, Double_t dz);
   TGeoTube(Double_t *params);
   // destructor
   virtual ~TGeoTube();
   // methods

   virtual Double_t      Capacity() const;
   static  Double_t      Capacity(Double_t rmin, Double_t rmax, Double_t dz);
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm);
   virtual void          ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize);
   static  void          ComputeNormalS(const Double_t *point, const Double_t *dir, Double_t *norm,
                                        Double_t rmin, Double_t rmax, Double_t dz);
   virtual Bool_t        Contains(const Double_t *point) const;
   virtual void          Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const;
   static  Double_t      DistFromInsideS(const Double_t *point, const Double_t *dir, Double_t rmin, Double_t rmax, Double_t dz);
   virtual Double_t      DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   static  Double_t      DistFromOutsideS(const Double_t *point, const Double_t *dir, Double_t rmin, Double_t rmax, Double_t dz);
   virtual Double_t      DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   static  void          DistToTube(Double_t rsq, Double_t nsq, Double_t rdotn, Double_t radius, Double_t &b, Double_t &delta);
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv,
                                Double_t start, Double_t step);
   virtual const char   *GetAxisName(Int_t iaxis) const;
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const {return 48;}
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const;
   virtual Double_t      GetRmin() const {return fRmin;}
   virtual Double_t      GetRmax() const {return fRmax;}
   virtual Double_t      GetDz() const   {return fDz;}
   Bool_t                HasRmin() const {return (fRmin>0)?kTRUE:kFALSE;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kTRUE;}
   virtual TBuffer3D    *MakeBuffer3D() const;
   virtual Double_t      Safety(const Double_t *point, Bool_t in=kTRUE) const;
   virtual void          Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const;
   static  Double_t      SafetyS(const Double_t *point, Bool_t in, Double_t rmin, Double_t rmax, Double_t dz, Int_t skipz=0);
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   void                  SetTubeDimensions(Double_t rmin, Double_t rmax, Double_t dz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoTube, 1)         // cylindrical tube class

};

class TGeoTubeSeg : public TGeoTube
{
protected:
   // data members
   Double_t              fPhi1;  // first phi limit
   Double_t              fPhi2;  // second phi limit
   // Transient trigonometric data
   Double_t              fS1;    // sin(phi1)
   Double_t              fC1;    // cos(phi1)
   Double_t              fS2;    // sin(phi2)
   Double_t              fC2;    // cos(phi2)
   Double_t              fSm;    // sin(0.5*(phi1+phi2))
   Double_t              fCm;    // cos(0.5*(phi1+phi2))
   Double_t              fCdfi;  // cos(0.5*(phi1-phi2))

   void                  InitTrigonometry();

public:
   // constructors
   TGeoTubeSeg();
   TGeoTubeSeg(Double_t rmin, Double_t rmax, Double_t dz,
               Double_t phi1, Double_t phi2);
   TGeoTubeSeg(const char * name, Double_t rmin, Double_t rmax, Double_t dz,
               Double_t phi1, Double_t phi2);
   TGeoTubeSeg(Double_t *params);
   // destructor
   virtual ~TGeoTubeSeg();
   // methods
   virtual void          AfterStreamer();
   virtual Double_t      Capacity() const;
   static  Double_t      Capacity(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2);
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm);
   virtual void          ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize);
   static  void          ComputeNormalS(const Double_t *point, const Double_t *dir, Double_t *norm,
                                        Double_t rmin, Double_t rmax, Double_t dz,
                                        Double_t c1, Double_t s1, Double_t c2, Double_t s2);
   virtual Bool_t        Contains(const Double_t *point) const;
   virtual void          Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   static  Double_t      DistFromInsideS(const Double_t *point, const Double_t *dir,Double_t rmin, Double_t rmax, Double_t dz,
                                    Double_t c1, Double_t s1, Double_t c2, Double_t s2, Double_t cm, Double_t sm, Double_t cdfi);
   virtual Double_t      DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   static  Double_t      DistFromOutsideS(const Double_t *point, const Double_t *dir, Double_t rmin, Double_t rmax, Double_t dz,
                                   Double_t c1, Double_t s1, Double_t c2, Double_t s2, Double_t cm, Double_t sm, Double_t cdfi);
   virtual Double_t      DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv,
                                Double_t start, Double_t step);
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const {return 56;}
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual Int_t         GetNmeshVertices() const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   Double_t              GetPhi1() const {return fPhi1;}
   Double_t              GetPhi2() const {return fPhi2;}
   virtual void          InspectShape() const;
   virtual TBuffer3D    *MakeBuffer3D() const;
   virtual Double_t      Safety(const Double_t *point, Bool_t in=kTRUE) const;
   virtual void          Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const;
   static  Double_t      SafetyS(const Double_t *point, Bool_t in, Double_t rmin, Double_t rmax, Double_t dz,
                                 Double_t phi1, Double_t phi2, Int_t skipz=0);
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   void                  SetTubsDimensions(Double_t rmin, Double_t rmax, Double_t dz,
                                       Double_t phi1, Double_t phi2);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoTubeSeg, 2)         // cylindrical tube segment class
};

class TGeoCtub : public TGeoTubeSeg
{
protected:
   // data members
   Double_t             fNlow[3];  // normal to lower cut plane
   Double_t             fNhigh[3]; // normal to higher cut plane

public:
   // constructors
   TGeoCtub();
   TGeoCtub(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
            Double_t lx, Double_t ly, Double_t lz, Double_t tx, Double_t ty, Double_t tz);
   TGeoCtub(const char *name, Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
            Double_t lx, Double_t ly, Double_t lz, Double_t tx, Double_t ty, Double_t tz);
   TGeoCtub(Double_t *params);
   // destructor
   virtual ~TGeoCtub();
   // methods
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm);
   virtual void          ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize);
   virtual Bool_t        Contains(const Double_t *point) const;
   virtual void          Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const;
   virtual Double_t      DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual Double_t      DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv,
                                Double_t start, Double_t step);
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const {return 98;}
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const;
   const Double_t       *GetNlow() const {return &fNlow[0];}
   const Double_t       *GetNhigh() const {return &fNhigh[0];}
   Double_t              GetZcoord(Double_t xc, Double_t yc, Double_t zc) const;
   virtual void          InspectShape() const;
   virtual Double_t      Safety(const Double_t *point, Bool_t in=kTRUE) const;
   virtual void          Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const;
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   void                  SetCtubDimensions(Double_t rmin, Double_t rmax, Double_t dz,
                                       Double_t phi1, Double_t phi2, Double_t lx, Double_t ly, Double_t lz,
                                       Double_t tx, Double_t ty, Double_t tz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;

   ClassDef(TGeoCtub, 1)         // cut tube segment class
};

#endif
