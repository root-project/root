// @(#)root/geom:$Name:$:$Id:$
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

class TGeoCompositeShape : public TGeoBBox
{
private :
// data members
   Int_t                 fNcomponents;    // number of components
   TList                *fShapes;         // list of TGeoShape
   TList                *fMatrices;       // list of matrices
   TGeoBoolCombinator   *fCombinator;     // boolean evaluator
// methods

public:
   // constructors
   TGeoCompositeShape();
   // destructor
   virtual ~TGeoCompositeShape();
   // methods
   void                  AddShape(TGeoShape *shape, TGeoMatrix *mat);
   void                  SetCombination(TGeoBoolCombinator *comb);

   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const {return kBig;}
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const {return kBig;}
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir) const;
   virtual void          Draw(Option_t *option);
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const {return 0;}
   virtual void          InspectShape() const;
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point) const;
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option) const;
   virtual void          SetDimensions(Double_t *param) {}
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const {;}

  ClassDef(TGeoCompositeShape, 1)         // boolean composite shape
};



#endif

