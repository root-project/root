/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - Wed 24 Oct 2001 05:20:43 PM CEST

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

class TGeoShape;

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
   TGeoBBox(Double_t *param);
   // destructor
   virtual ~TGeoBBox();
   // methods
   virtual Int_t         GetByteCount() {return 36;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;

   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);
   virtual Bool_t        CouldBeCrossed(Double_t *point, Double_t *dir);
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          Draw(Option_t *option);

   virtual Double_t      GetDX()  {return fDX;}
   virtual Double_t      GetDY()  {return fDY;}
   virtual Double_t      GetDZ()  {return fDZ;}
   virtual Double_t     *GetOrigin() {return &fOrigin[0];}
   
   virtual void          InspectShape();
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   void                  SetBoxDimensions(Double_t dx, Double_t dy, Double_t dz, Double_t *origin=0);
   virtual void          SetDimensions(Double_t *param);
   void                  SetBoxPoints(Double_t *buff);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoBBox, 1)         // box primitive
};

#endif
