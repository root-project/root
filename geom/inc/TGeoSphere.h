// @(#)root/geom:$Name:  $:$Id: TGeoSphere.h,v 1.2 2002/07/10 19:24:16 brun Exp $
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoSphere
#define ROOT_TGeoSphere

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

/*************************************************************************
 * TGeoSphere - spherical shell class. It takes 6 parameters : 
 *           - inner and outer radius Rmin, Rmax
 *           - the theta limits Tmin, Tmax
 *           - the phi limits Pmin, Pmax (the sector in phi is considered
 *             starting from Pmin to Pmax counter-clockwise
 *
 *************************************************************************/

class TGeoSphere : public TGeoBBox
{
protected :
// data members
   Int_t                 fNz;     // number of z planes for drawing
   Int_t                 fNseg;   // number of segments for drawing
   Double_t              fRmin;   // inner radius
   Double_t              fRmax;   // outer radius
   Double_t              fTheta1; // lower theta limit
   Double_t              fTheta2; // higher theta limit
   Double_t              fPhi1;   // lower phi limit
   Double_t              fPhi2;   // higher phi limit
// methods

public:
   // constructors
   TGeoSphere();
   TGeoSphere(Double_t rmin, Double_t rmax, Double_t theta1=0, Double_t theta2=180,
              Double_t phi1=0, Double_t phi2=360);
   TGeoSphere(Double_t *param, Int_t nparam=6);
   // destructor
   virtual ~TGeoSphere();
   // methods
   virtual Int_t         GetByteCount() const {return 42;}
   void                  SetSphDimensions(Double_t rmin, Double_t rmax, Double_t theta1,
                                       Double_t theta2, Double_t phi1, Double_t phi2);
   void                  SetNumberOfDivisions(Int_t p);
   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const;
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const;
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step) {return voldiv;}
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Double_t step);

   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const {return 0;}
   Int_t                 GetNumberOfDivisions() const {return fNseg;}
   Int_t                 GetNz() const   {return fNz;}
   virtual Double_t      GetRmin() const {return fRmin;}
   virtual Double_t      GetRmax() const {return fRmax;}
   Double_t              GetTheta1() const {return fTheta1;}
   Double_t              GetTheta2() const {return fTheta2;}
   Double_t              GetPhi1() const {return fPhi1;}
   Double_t              GetPhi2() const {return fPhi2;}

   virtual void          InspectShape() const;
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point) const;
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option) const;
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoSphere, 1)         // sphere class
};

#endif
