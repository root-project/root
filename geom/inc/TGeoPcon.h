// @(#)root/geom:$Name:  $:$Id: TGeoPcon.h,v 1.2 2002/07/10 19:24:16 brun Exp $
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
   virtual Bool_t        Contains(Double_t *point) const;

   virtual void          DefineSection(Int_t snum, Double_t z, Double_t rmin, Double_t rmax);
   
   virtual Int_t         GetByteCount() const {return 60+12*fNz;}
   Double_t              GetPhi1() const {return fPhi1;}
   Double_t              GetDphi() const {return fDphi;}
   Int_t                 GetNz() const   {return fNz;}
   virtual Int_t         GetNsegments() const;
   Double_t             *GetRmin() const {return fRmin;}
   Double_t             *GetRmax() const {return fRmax;}
   Double_t             *GetZ() const    {return fZ;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const {return 0;}
     
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const;
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) const;
   Double_t              DistToSegZ(Double_t *point, Double_t *dir, Int_t &iz, Double_t c1, Double_t s1,
                                    Double_t c2, Double_t s2, Double_t cfio, Double_t sfio, Double_t cdfi) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Double_t step);
   virtual void          InspectShape() const;
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point) const;
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option) const;
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoPcon, 1)         // polycone class 
};

#endif
