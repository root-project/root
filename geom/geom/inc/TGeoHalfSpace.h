// @(#) :$Id$
// Author: Mihaela Gheata   03/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoHalfSpace
#define ROOT_TGeoHalfSpace

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoHalfSpace - A half-space defined by:                               // 
//            p[3] - an arbitrary point on the plane                      //
//            n[3] - normal at the plane in point P                       //
//    A half-space is not really a shape, because it is infinite. The     //
//    normal vector points "outside" the half-space                       //
//                                                                        //
////////////////////////////////////////////////////////////////////////////


class TGeoHalfSpace : public TGeoBBox
{
private:
   Double_t              fP[3];
   Double_t              fN[3];
public:
   // constructors
   TGeoHalfSpace();
   TGeoHalfSpace(const char *name, Double_t *p, Double_t *n);
   TGeoHalfSpace(Double_t *params);
   // destructor
   virtual ~TGeoHalfSpace();
   // methods
   virtual Double_t      Capacity() const {return 0.;}
   virtual void          ComputeBBox() {;}
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual Double_t     *GetPoint()    {return fP;}
   virtual Double_t     *GetNorm()     {return fN;}
   virtual void          GetBoundingCylinder(Double_t * /*param*/) const {;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const {return 0;}
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const {return 0;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kFALSE;}
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t * /*points*/) const {;}
   virtual void          SetPoints(Float_t * /*points*/) const {;}
   virtual void          Sizeof3D() const {;}

   ClassDef(TGeoHalfSpace, 1)         // half-space class
};


#endif
