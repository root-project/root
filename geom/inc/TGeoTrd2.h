// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTrd2
#define ROOT_TGeoTrd2

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

  
 /*************************************************************************
 * TGeoTrd2 - a trapezoid with both x and y lengths varying with z. It 
 *   has 5 parameters, the half lengths in x at -dz and +dz, the half
 *  lengths in y at -dz and +dz, and the half length in z (dz).
 *
 *************************************************************************/

class TGeoTrd2 : public TGeoBBox
{
protected:
   // data members
   Double_t              fDx1; // half length in X at lower Z surface (-dz)
   Double_t              fDx2; // half length in X at higher Z surface (+dz)
   Double_t              fDy1; // half length in Y at lower Z surface (-dz)
   Double_t              fDy2; // half length in Y at higher Z surface (+dz)
   Double_t              fDz;  // half length in Z
   
public:
   // constructors
   TGeoTrd2();
   TGeoTrd2(Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2, Double_t dz);
   TGeoTrd2(Double_t *params);
   // destructor
   virtual ~TGeoTrd2();
   // methods
   virtual Int_t         GetByteCount() const {return 56;}
   Double_t              GetDx1() const {return fDx1;}
   Double_t              GetDx2() const {return fDx2;}
   Double_t              GetDy1() const {return fDy1;}
   Double_t              GetDy2() const {return fDy2;}
   Double_t              GetDz() const  {return fDz;}

   virtual Bool_t        Contains(Double_t *point) const;
   virtual void          ComputeBBox();
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const;
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const;
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir) const;
   virtual void          Draw(Option_t *option);
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;
   void                  GetVisibleCorner(Double_t *point, Double_t *vertex, Double_t *normals) const;
   void                  GetOppositeCorner(Double_t *point, Int_t inorm, Double_t *vertex, Double_t *normals) const;
   virtual void          InspectShape() const;
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point) const;
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option) const;
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   void                  SetVertex(Double_t *vertex) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoTrd2, 1)         // TRD2 shape class
};

#endif
