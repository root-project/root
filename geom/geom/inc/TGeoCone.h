// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoCone
#define ROOT_TGeoCone

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif



////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoCone - conical tube  class. It has 5 parameters :                  //
//            dz - half length in z                                       //
//            Rmin1, Rmax1 - inside and outside radii at -dz              //
//            Rmin2, Rmax2 - inside and outside radii at +dz              //
//                                                                        //
////////////////////////////////////////////////////////////////////////////


class TGeoCone : public TGeoBBox
{
protected :
// data members
   Double_t              fDz;    // half length
   Double_t              fRmin1; // inner radius at -dz
   Double_t              fRmax1; // outer radius at -dz
   Double_t              fRmin2; // inner radius at +dz
   Double_t              fRmax2; // outer radius at +dz
// methods
public:
   // constructors
   TGeoCone();
   TGeoCone(Double_t dz, Double_t rmin1, Double_t rmax1,
            Double_t rmin2, Double_t rmax2);
   TGeoCone(const char *name, Double_t dz, Double_t rmin1, Double_t rmax1,
            Double_t rmin2, Double_t rmax2);
   TGeoCone(Double_t *params);
   // destructor
   virtual ~TGeoCone();
   // methods

   virtual Double_t      Capacity() const;
   static  Double_t      Capacity(Double_t dz, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2);
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm);
   virtual void          ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize);
   static  void          ComputeNormalS(const Double_t *point, const Double_t *dir, Double_t *norm,
                                        Double_t dz, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2);
   virtual Bool_t        Contains(const Double_t *point) const;
   virtual void          Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   static  void          DistToCone(const Double_t *point, const Double_t *dir, Double_t dz, Double_t r1, Double_t r2, Double_t &b, Double_t &delta);
   static  Double_t      DistFromInsideS(const Double_t *point, const Double_t *dir, Double_t dz,
                                    Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2);
   virtual Double_t      DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   static  Double_t      DistFromOutsideS(const Double_t *point, const Double_t *dir, Double_t dz,
                                   Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2);
   virtual Double_t      DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv,
                                Double_t start, Double_t step);

   virtual const char   *GetAxisName(Int_t iaxis) const;
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual Int_t         GetByteCount() const {return 56;}
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Double_t      GetDz() const    {return fDz;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const;
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const;
   virtual Double_t      GetRmin1() const {return fRmin1;}
   virtual Double_t      GetRmax1() const {return fRmax1;}
   virtual Double_t      GetRmin2() const {return fRmin2;}
   virtual Double_t      GetRmax2() const {return fRmax2;}

   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kTRUE;}
   virtual TBuffer3D    *MakeBuffer3D() const;
   virtual Double_t      Safety(const Double_t *point, Bool_t in=kTRUE) const;
   virtual void          Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const;
   static  Double_t      SafetyS(const Double_t *point, Bool_t in, Double_t dz, Double_t rmin1, Double_t rmax1,
                                 Double_t rmin2, Double_t rmax2, Int_t skipz=0);
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   void                  SetConeDimensions(Double_t dz, Double_t rmin1, Double_t rmax1,
                                       Double_t rmin2, Double_t rmax2);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buffer) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoCone, 1)         // conical tube class

};

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoConeSeg - a phi segment of a conical tube. Has 7 parameters :      //
//            - the same 5 as a cone;                                     //
//            - first phi limit (in degrees)                              //
//            - second phi limit                                          //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoConeSeg : public TGeoCone
{
protected:
   // data members
   Double_t              fPhi1;  // first phi limit
   Double_t              fPhi2;  // second phi limit
   // Transient trigonometric data
   Double_t              fS1;    //!sin(phi1)
   Double_t              fC1;    //!cos(phi1)
   Double_t              fS2;    //!sin(phi2)
   Double_t              fC2;    //!cos(phi2)
   Double_t              fSm;    //!sin(0.5*(phi1+phi2))
   Double_t              fCm;    //!cos(0.5*(phi1+phi2))
   Double_t              fCdfi;  //!cos(0.5*(phi1-phi2))

   void                  InitTrigonometry();

public:
   // constructors
   TGeoConeSeg();
   TGeoConeSeg(Double_t dz, Double_t rmin1, Double_t rmax1,
               Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2);
   TGeoConeSeg(const char *name, Double_t dz, Double_t rmin1, Double_t rmax1,
               Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2);
   TGeoConeSeg(Double_t *params);
   // destructor
   virtual ~TGeoConeSeg();
   // methods
   virtual void          AfterStreamer();
   virtual Double_t      Capacity() const;
   static  Double_t      Capacity(Double_t dz, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2);
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm);
   virtual void          ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize);
   static  void          ComputeNormalS(const Double_t *point, const Double_t *dir, Double_t *norm,
                                        Double_t dz, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2,
                                        Double_t c1, Double_t s1, Double_t c2, Double_t s2);
   virtual Bool_t        Contains(const Double_t *point) const;
   virtual void          Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const;

   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   static  Double_t      DistToCons(const Double_t *point, const Double_t *dir, Double_t r1, Double_t z1, Double_t r2, Double_t z2, Double_t phi1, Double_t phi2);
   static  Double_t      DistFromInsideS(const Double_t *point, const Double_t *dir, Double_t dz, Double_t rmin1, Double_t rmax1,
                                   Double_t rmin2, Double_t rmax2, Double_t c1, Double_t s1, Double_t c2, Double_t s2, Double_t cm, Double_t sm, Double_t cdfi);
   virtual Double_t      DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   static  Double_t      DistFromOutsideS(const Double_t *point, const Double_t *dir, Double_t dz, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2,
                                   Double_t c1, Double_t s1, Double_t c2, Double_t s2, Double_t cm, Double_t sm, Double_t cdfi);
   virtual Double_t      DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv,
                                Double_t start, Double_t step);
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const {return 64;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const;
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const;
   Double_t              GetPhi1() const {return fPhi1;}
   Double_t              GetPhi2() const {return fPhi2;}
   virtual void          InspectShape() const;
   virtual TBuffer3D    *MakeBuffer3D() const;
   virtual Double_t      Safety(const Double_t *point, Bool_t in=kTRUE) const;
   virtual void          Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const;
   static  Double_t      SafetyS(const Double_t *point, Bool_t in, Double_t dz, Double_t rmin1, Double_t rmax1,
                                 Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2, Int_t skipz=0);
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   void                  SetConsDimensions(Double_t dz, Double_t rmin1, Double_t rmax1,
                                       Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buffer) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoConeSeg, 1)         // conical tube segment class
};

#endif
