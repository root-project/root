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

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        
// TGeoParaboloid - Paraboloid  class. A paraboloid is the solid bounded by
//            the following surfaces:
//            - 2 planes parallel with XY cutting the Z axis at Z=-dz and Z=+dz
//            - the surface of revolution of a parabola described by:
//                 z = a*(x*x + y*y) + b
//       The parameters a and b are automatically computed from:
//            - rlo - the radius of the circle of intersection between the 
//              parabolic surface and the plane z = -dz
//            - rhi - the radius of the circle of intersection between the 
//              parabolic surface and the plane z = +dz
//         | -dz = a*rlo*rlo + b
//         |  dz = a*rhi*rhi + b      where: rlo != rhi, both >= 0
//                                                                        
////////////////////////////////////////////////////////////////////////////


class TGeoParaboloid : public TGeoBBox
{
private:
   Double_t              fRlo;                  // radius at z=-dz
   Double_t              fRhi;                  // radius at z=+dz
   Double_t              fDz;                   // range on Z axis [-dz, dz]
   Double_t              fA;                    // quadratic coeff.
   Double_t              fB;                    // Z value of parabola at x=y=0
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
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   Double_t              DistToParaboloid(Double_t *point, Double_t *dir, Bool_t in) const;
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
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
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   void                  SetParaboloidDimensions(Double_t rlo, Double_t rhi, Double_t dz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoParaboloid, 1)         // paraboloid class

};


#endif
