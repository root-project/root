/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Mihala Gheata - Fri 05 Jul 2002 03:05:49 PM CEST

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
   TGeoEltu(Double_t *params);
   // destructor
   virtual ~TGeoEltu();
   // methods
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;

   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);

   virtual Double_t      GetA()    {return fRmin;}
   virtual Double_t      GetB()    {return fRmax;}
   
   virtual void          InspectShape();
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   void                  SetEltuDimensions(Double_t a, Double_t b, Double_t dz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;

  ClassDef(TGeoEltu, 1)         // elliptical tube class

};


#endif
