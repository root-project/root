/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - Wed 24 Oct 2001 05:20:43 PM CEST
// TGeoShape::Contains implemented by Mihaela Gheata

#ifndef ROOT_TGeoPcon
#define ROOT_TGeoPcon

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

  
/*************************************************************************
 * TGeoPcon - a composite polycone. It has at least 9 parameters :
 *            - the lower phi limit;
 *            - the range in phi;
 *            - the number of z planes (at least two) where the inner/outer 
 *              radii are changing;
 *            - z coordinate, inner and outer radius for each z plane
 *
 *************************************************************************/

class TGeoPcon : public TGeoBBox
{
protected:
   // data members
   Int_t                 fNz;    // number of z planes (at least two)
   Double_t              fPhi1;  // lower phi limit 
   Double_t              fDphi;  // phi range
   Double_t             *fRmin;  //[fNz] pointer to array of inner radii 
   Double_t             *fRmax;  //[fNz] pointer to array of outer radii 
   Double_t             *fZ;     //[fNz] pointer to array of Z planes positions 
public:
   // constructors
   TGeoPcon();
   TGeoPcon(Double_t phi, Double_t dphi, Int_t nz);
   TGeoPcon(Double_t *params);
   // destructor
   virtual ~TGeoPcon();
   // methods
   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);

   virtual void          DefineSection(Int_t snum, Double_t z, Double_t rmin, Double_t rmax);
   
   virtual Int_t         GetByteCount() {return 60+12*fNz;}
   Double_t              GetPhi1() {return fPhi1;}
   Double_t              GetDphi() {return fDphi;}
   Int_t                 GetNz()   {return fNz;}
   virtual Int_t         GetNsegments();
   Double_t             *GetRmin() {return fRmin;}
   Double_t             *GetRmax() {return fRmax;}
   Double_t             *GetZ()    {return fZ;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const {return 0;}
     
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   Double_t              DistToSegZ(Double_t *point, Double_t *dir, Int_t &iz, Double_t c1, Double_t s1,
                                    Double_t c2, Double_t s2, Double_t cfio, Double_t sfio, Double_t cdfi);
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          Draw(Option_t *option);
   virtual void          InspectShape();
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoPcon, 1)         // polycone class 
};

#endif
