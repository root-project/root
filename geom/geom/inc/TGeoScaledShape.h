// @(#)root/geom:$Id$
// Author: Andrei Gheata   26/09/05
   
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoScaledShape
#define ROOT_TGeoScaledShape

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoScaledShape - A scaled shape. Has a pointer to a shape and to a    //
//   TGeoScale.                                                           //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoScale;
class TGeoShape;

class TGeoScaledShape : public TGeoBBox
{
protected :
// data members
   TGeoShape            *fShape;     // pointer to an existing shape
   TGeoScale            *fScale;     // pointer to a scale transformation
// methods
public:
   // constructors
   TGeoScaledShape();
   TGeoScaledShape(const char *name, TGeoShape *shape, TGeoScale *scale);
   TGeoScaledShape(TGeoShape *shape, TGeoScale *scale);
   // destructor
   virtual ~TGeoScaledShape();
   // methods
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const {return fShape->GetNmeshVertices();}
   TGeoShape            *GetShape() const {return fShape;}
   TGeoScale            *GetScale() const {return fScale;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsAssembly() const;
   virtual Bool_t        IsCylType() const {return fShape->IsCylType();}
   virtual Bool_t        IsReflected() const;
   virtual TBuffer3D    *MakeBuffer3D() const;
   static  TGeoShape    *MakeScaledShape(const char *name, TGeoShape *shape, TGeoScale *scale);
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   void                  SetScale(TGeoScale *scale) {fScale = scale;}
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buffer) const;

   ClassDef(TGeoScaledShape, 1)         // a scaled shape
};

#endif
