// @(#)root/geom:$Name:  $:$Id: TGeoBBox.h,v 1.9 2003/07/31 20:19:31 brun Exp $
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

/*************************************************************************
 * TGeoBBox - box class. All shape primitives inherit from this, their 
 *   constructor filling automatically the parameters of the box that bounds
 *   the given shape. Defined by 6 parameters :
 *      fDX, fDY, fDZ - half lengths on X, Y and Z axis
 *      fOrigin[3]    - position of box origin
 *
 *************************************************************************/

class TGeoBBox : public TGeoShape
{
protected :
// data members
   Double_t              fDX;        // X half-length
   Double_t              fDY;        // Y half-length
   Double_t              fDZ;        // Z half-length
   Double_t              fOrigin[3]; // box origin
// methods

public:
   // constructors
   TGeoBBox();
   TGeoBBox(Double_t dx, Double_t dy, Double_t dz, Double_t *origin=0);
   TGeoBBox(const char *name, Double_t dx, Double_t dy, Double_t dz, Double_t *origin=0);
   TGeoBBox(Double_t *param);
   // destructor
   virtual ~TGeoBBox();
   // methods
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Bool_t        CouldBeCrossed(Double_t *point, Double_t *dir) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=kBig, Double_t *safe=0) const;
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=kBig, Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual const char   *GetAxisName(Int_t iaxis) const;
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual Int_t         GetByteCount() const {return 36;}
   virtual Int_t         GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual Double_t      GetDX() const  {return fDX;}
   virtual Double_t      GetDY() const  {return fDY;}
   virtual Double_t      GetDZ() const  {return fDZ;}
   virtual const Double_t *GetOrigin() const {return fOrigin;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kFALSE;}
   virtual Bool_t        IsValidBox() const {return ((fDX<0)||(fDY<0)||(fDZ<0))?kFALSE:kTRUE;}
   virtual Bool_t        IsNullBox() const {return ((fDX==0)&&(fDY==0)&&(fDZ==0))?kTRUE:kFALSE;}
   virtual void         *Make3DBuffer(const TGeoVolume *vol) const;
   virtual void          Paint(Option_t *option);
   virtual void          PaintNext(TGeoHMatrix *glmat, Option_t *option);
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   void                  SetBoxDimensions(Double_t dx, Double_t dy, Double_t dz, Double_t *origin=0);
   virtual void          SetDimensions(Double_t *param);
   void                  SetBoxPoints(Double_t *buff) const;
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoBBox, 1)         // box primitive
};

#endif
