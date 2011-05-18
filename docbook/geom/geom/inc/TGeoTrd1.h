// @(#)root/geom:$Id$
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoTrd1
#define ROOT_TGeoTrd1

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoTrd1 - a trapezoid with only x length varying with z. It has 4     //
//   parameters, the half length in x at the low z surface, that at the   //
//   high z surface, the half length in y, and in z                       //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoTrd1 : public TGeoBBox
{
protected:
   // data members
   Double_t              fDx1; // half length in X at lower Z surface (-dz)
   Double_t              fDx2; // half length in X at higher Z surface (+dz)
   Double_t              fDy;  // half length in Y
   Double_t              fDz;  // half length in Z

   // methods

public:
   // constructors
   TGeoTrd1();
   TGeoTrd1(Double_t dx1, Double_t dx2, Double_t dy, Double_t dz);
   TGeoTrd1(const char *name, Double_t dx1, Double_t dx2, Double_t dy, Double_t dz);
   TGeoTrd1(Double_t *params);
   // destructor
   virtual ~TGeoTrd1();
   // methods

   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual Int_t         GetByteCount() const {return 52;}
   Double_t              GetDx1() const {return fDx1;}
   Double_t              GetDx2() const {return fDx2;}
   Double_t              GetDy() const  {return fDy;}
   Double_t              GetDz() const  {return fDz;}
   virtual Int_t         GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   void                  GetVisibleCorner(Double_t *point, Double_t *vertex, Double_t *normals) const;
   void                  GetOppositeCorner(Double_t *point, Int_t inorm, Double_t *vertex, Double_t *normals) const;
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kFALSE;}
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   void                  SetVertex(Double_t *vertex) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoTrd1, 1)         // TRD1 shape class
};

#endif
