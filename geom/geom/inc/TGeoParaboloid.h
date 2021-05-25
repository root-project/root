// @(#)root/geom:$Id$
// Author: Mihaela Gheata   20/06/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoParaboloid
#define ROOT_TGeoParaboloid

#include "TGeoBBox.h"

class TGeoParaboloid : public TGeoBBox
{
private:
   Double_t              fRlo;                  // radius at z=-dz
   Double_t              fRhi;                  // radius at z=+dz
   Double_t              fDz;                   // range on Z axis [-dz, dz]
   Double_t              fA;                    // quadratic coeff.
   Double_t              fB;                    // Z value of parabola at x=y=0

   TGeoParaboloid(const TGeoParaboloid&) = delete;
   TGeoParaboloid& operator=(const TGeoParaboloid&) = delete;

public:
   // constructors
   TGeoParaboloid();
   TGeoParaboloid(Double_t rlo, Double_t rhi, Double_t dz);
   TGeoParaboloid(const char *name, Double_t rlo, Double_t rhi, Double_t dz);
   TGeoParaboloid(Double_t *params);
   // destructor
   virtual ~TGeoParaboloid();
   // methods
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm);
   virtual void          ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize);
   virtual Bool_t        Contains(const Double_t *point) const;
   virtual void          Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   Double_t              DistToParaboloid(const Double_t *point, const Double_t *dir, Bool_t in) const;
   virtual Double_t      DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual Double_t      DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv,
                                Double_t start, Double_t step);
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   Double_t              GetRlo() const    {return fRlo;}
   Double_t              GetRhi() const    {return fRhi;}
   Double_t              GetDz() const     {return fDz;}

   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const;
   virtual Bool_t        GetPointsOnSegments(Int_t /*npoints*/, Double_t * /*array*/) const {return kFALSE;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kTRUE;}
   virtual TBuffer3D    *MakeBuffer3D() const;
   virtual Double_t      Safety(const Double_t *point, Bool_t in=kTRUE) const;
   virtual void          Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const;
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   void                  SetParaboloidDimensions(Double_t rlo, Double_t rhi, Double_t dz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoParaboloid, 1)         // paraboloid class

};


#endif
