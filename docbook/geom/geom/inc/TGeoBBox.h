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

#ifndef ROOT_TGeoShape
#include "TGeoShape.h"
#endif

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// TGeoBBox - box class. All shape primitives inherit from this, their       //
//   constructor filling automatically the parameters of the box that bounds //
//   the given shape. Defined by 6 parameters :                              //
//      fDX, fDY, fDZ - half lengths on X, Y and Z axis                      //
//      fOrigin[3]    - position of box origin                               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

class TGeoBBox : public TGeoShape
{
protected :
// data members
   Double_t              fDX;        // X half-length
   Double_t              fDY;        // Y half-length
   Double_t              fDZ;        // Z half-length
   Double_t              fOrigin[3]; // box origin
// methods
   virtual void FillBuffer3D(TBuffer3D & buffer, Int_t reqSections, Bool_t localFrame) const;
public:
   // constructors
   TGeoBBox();
   TGeoBBox(Double_t dx, Double_t dy, Double_t dz, Double_t *origin=0);
   TGeoBBox(const char *name, Double_t dx, Double_t dy, Double_t dz, Double_t *origin=0);
   TGeoBBox(Double_t *param);
   // destructor
   virtual ~TGeoBBox();
   // methods
   static  Bool_t        AreOverlapping(const TGeoBBox *box1, const TGeoMatrix *mat1, const TGeoBBox *box2, const TGeoMatrix *mat2);
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   static  Bool_t        Contains(const Double_t *point, Double_t dx, Double_t dy, Double_t dz, const Double_t *origin);
   virtual Bool_t        CouldBeCrossed(Double_t *point, Double_t *dir) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   static  Double_t      DistFromInside(const Double_t *point,const Double_t *dir, 
                                   Double_t dx, Double_t dy, Double_t dz, const Double_t *origin, Double_t stepmax=TGeoShape::Big());
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   static  Double_t      DistFromOutside(const Double_t *point,const Double_t *dir, 
                                   Double_t dx, Double_t dy, Double_t dz, const Double_t *origin, Double_t stepmax=TGeoShape::Big());
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual const char   *GetAxisName(Int_t iaxis) const;
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const {return 36;}
   virtual Double_t      GetFacetArea(Int_t index=0) const;
   virtual Bool_t        GetPointsOnFacet(Int_t index, Int_t npoints, Double_t *array) const;
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const;
   virtual Int_t         GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const {return 8;}
   virtual Double_t      GetDX() const  {return fDX;}
   virtual Double_t      GetDY() const  {return fDY;}
   virtual Double_t      GetDZ() const  {return fDZ;}
   virtual const Double_t *GetOrigin() const {return fOrigin;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kFALSE;}
   virtual Bool_t        IsValidBox() const {return ((fDX<0)||(fDY<0)||(fDZ<0))?kFALSE:kTRUE;}
   virtual Bool_t        IsNullBox() const {return ((fDX<1.E-16)&&(fDY<1.E-16)&&(fDZ<1.E-16))?kTRUE:kFALSE;}
   virtual TBuffer3D    *MakeBuffer3D() const;
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   void                  SetBoxDimensions(Double_t dx, Double_t dy, Double_t dz, Double_t *origin=0);
   virtual void          SetDimensions(Double_t *param);
   void                  SetBoxPoints(Double_t *points) const;
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buffer) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoBBox, 1)         // box primitive
};

#endif
