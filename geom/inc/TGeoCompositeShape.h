// @(#)root/geom:$Name:  $:$Id: TGeoCompositeShape.h,v 1.13 2004/04/22 14:07:14 brun Exp $
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoCompositeShape
#define ROOT_TGeoCompositeShape

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif
    
 /*************************************************************************
 * TGeoCompositeShape - composite shape class. A composite shape contains
 *   a list of primitive shapes, the list of coresponding transformations
 *   and a boolean finder handling boolean operations among components.
 *   
 *
 *************************************************************************/

class TGeoBoolNode;

class TGeoCompositeShape : public TGeoBBox
{
private :
// data members
   TGeoBoolNode         *fNode;             // top boolean node
   
public:
   // constructors
   TGeoCompositeShape();
   TGeoCompositeShape(const char *name, const char *expression);
   TGeoCompositeShape(const char *expression);
   TGeoCompositeShape(const char *name, TGeoBoolNode *node);
   // destructor
   virtual ~TGeoCompositeShape();
   // methods
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   TGeoBoolNode         *GetBoolNode() const {return fNode;}
   virtual void          GetBoundingCylinder(Double_t * /*param*/) const {;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const {return 0;}
   virtual Int_t         GetNmeshVertices() const;
   virtual void          InspectShape() const;
   virtual Bool_t        IsComposite() const {return kTRUE;}
   virtual Bool_t        IsCylType() const {return kFALSE;}
   virtual void         *Make3DBuffer(const TGeoVolume *vol) const;
   void                  MakeNode(const char *expression);
   virtual void          Paint(Option_t *option);
   virtual void          PaintNext(TGeoHMatrix *glmat, Option_t *option);
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SetDimensions(Double_t * /*param*/) {;}
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoCompositeShape, 1)         // boolean composite shape
};



#endif

