// @(#)root/geom:$Name:  $:$Id: TGeoEltu.h,v 1.6 2003/03/14 11:49:02 brun Exp $
// Author: Mihaela Gheata   05/06/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoEltu
#define ROOT_TGeoEltu

#ifndef ROOT_TGeoTube
#include "TGeoTube.h"
#endif

/*************************************************************************
 * TGeoEltu - elliptical tube  class. An elliptical tube has 3 parameters :
 *            A - semi-axis of the ellipse along x
 *            B - semi-axis of the ellipse along y
 *            dz - half length in z
 *
 *************************************************************************/


class TGeoEltu : public TGeoTube
{
public:
   // constructors
   TGeoEltu();
   TGeoEltu(Double_t a, Double_t b, Double_t dz);
   TGeoEltu(const char *name, Double_t a, Double_t b, Double_t dz);
   TGeoEltu(Double_t *params);
   // destructor
   virtual ~TGeoEltu();
   // methods
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const;
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual Double_t      GetA() const    {return fRmin;}
   virtual Double_t      GetB() const    {return fRmax;}
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kTRUE;}
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   void                  SetEltuDimensions(Double_t a, Double_t b, Double_t dz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;

  ClassDef(TGeoEltu, 1)         // elliptical tube class

};


#endif
