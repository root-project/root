/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - Wed 24 Oct 2001 05:20:43 PM CEST
// TGeoShape::Contains implemented by Mihaela Gheata

#ifndef ROOT_TGeoCone
#define ROOT_TGeoCone

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif



/*************************************************************************
 * TGeoCone - conical tube  class. It has 5 parameters :
 *            dz - half length in z
 *            Rmin1, Rmax1 - inside and outside radii at -dz
 *            Rmin2, Rmax2 - inside and outside radii at +dz
 *
 *************************************************************************/


class TGeoCone : public TGeoBBox
{
protected :
// data members
   Double_t              fDz;    // half length
   Double_t              fRmin1; // inner radius at -dz
   Double_t              fRmax1; // outer radius at -dz
   Double_t              fRmin2; // inner radius at +dz
   Double_t              fRmax2; // outer radius at +dz
// methods

public:
   // constructors
   TGeoCone();
   TGeoCone(Double_t dz, Double_t rmin1, Double_t rmax1,
            Double_t rmin2, Double_t rmax2);
   TGeoCone(Double_t *params);
   // destructor
   virtual ~TGeoCone();
   // methods

   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   static  Double_t      DistToOutS(Double_t *point, Double_t *dir, Int_t iact,Double_t step, Double_t *safe,
                                    Double_t dz,Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   static  Double_t      DistToInS(Double_t *point, Double_t *dir, Double_t rmin1, Double_t rmax1,Double_t rmin2, Double_t rmax2, 
                                   Double_t dz, Double_t ro1, Double_t tg1, Double_t cr1, Double_t zv1,
                                   Double_t ro2, Double_t tg2, Double_t cr2, Double_t zv2, Double_t r2, Double_t rin, Double_t rout);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          Draw(Option_t *option);

   virtual Int_t         GetByteCount() {return 56;}
   virtual Double_t      GetDz()    {return fDz;}
   virtual Double_t      GetRmin1() {return fRmin1;}
   virtual Double_t      GetRmax1() {return fRmax1;}
   virtual Double_t      GetRmin2() {return fRmin2;}
   virtual Double_t      GetRmax2() {return fRmax2;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;
   
   virtual void          InspectShape();
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   void                  SetConeDimensions(Double_t dz, Double_t rmin1, Double_t rmax1,
                                       Double_t rmin2, Double_t rmax2);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoCone, 1)         // conical tube class

};

/*************************************************************************
 * TGeoConeSeg - a phi segment of a conical tube. Has 7 parameters :
 *            - the same 5 as a cone;
 *            - first phi limit (in degrees)
 *            - second phi limit 
 *
 *************************************************************************/

class TGeoConeSeg : public TGeoCone
{
protected:
   // data members
   Double_t              fPhi1;  // first phi limit 
   Double_t              fPhi2;  // second phi limit 
    
   static Double_t       DistToPhiMin(Double_t *point, Double_t *dir, Double_t s1, Double_t c1,
                                      Double_t s2, Double_t c2, Double_t sm, Double_t cm);   
public:
   // constructors
   TGeoConeSeg();
   TGeoConeSeg(Double_t dz, Double_t rmin1, Double_t rmax1,
               Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2);
   TGeoConeSeg(Double_t *params);
   // destructor
   virtual ~TGeoConeSeg();
   // methods
   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);

   virtual Int_t         GetByteCount() {return 64;}
   Double_t              GetPhi1() {return fPhi1;}
   Double_t              GetPhi2() {return fPhi2;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;
   
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   static  Double_t      DistToOutS(Double_t *point, Double_t *dir, Int_t iact,Double_t step, Double_t *safe,
                                    Double_t dz,Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2,
                                    Double_t phi1, Double_t phi2);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   static  Double_t      DistToInS(Double_t *point, Double_t *dir, Double_t rmin1, Double_t rmax1,Double_t rmin2, Double_t rmax2, 
                                   Double_t dz, Double_t ro1, Double_t tg1, Double_t cr1, Double_t zv1,
                                   Double_t ro2, Double_t tg2, Double_t cr2, Double_t zv2, Double_t r2, Double_t rin, Double_t rout,
                                   Double_t c1, Double_t s1, Double_t c2, Double_t s2, Double_t cfio, Double_t sfio, Double_t cdfi);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          Draw(Option_t *option);
   virtual void          InspectShape();
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   void                  SetConsDimensions(Double_t dz, Double_t rmin1, Double_t rmax1,
                                       Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoConeSeg, 1)         // conical tube segment class 
};

#endif
