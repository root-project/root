/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - Wed 24 Oct 2001 05:20:43 PM CEST
// TGeoShape::Contains implemented by Mihaela Gheata

#ifndef ROOT_TGeoPgon
#define ROOT_TGeoPgon

#ifndef ROOT_TGeoPcon
#include "TGeoPcon.h"
#endif

  
/*************************************************************************
 * TGeoPgon - a polygone. It has at least 10 parameters :
 *            - the lower phi limit;
 *            - the range in phi;
 *            - the number of edges on each z plane;
 *            - the number of z planes (at least two) where the inner/outer 
 *              radii are changing;
 *            - z coordinate, inner and outer radius for each z plane
 *
 *************************************************************************/

class TGeoPgon : public TGeoPcon
{
protected:
   // data members
   Int_t                 fNedges;    // number of z planes (at least two)
public:
   // constructors
   TGeoPgon();
   TGeoPgon(Double_t phi, Double_t dphi, Int_t nedges, Int_t nz);
   TGeoPgon(Double_t *params);
   // destructor
   virtual ~TGeoPgon();
   // methods
   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);

   virtual Int_t         GetByteCount() {return 64+12*fNz;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const {return 0;}
   Int_t                 GetNedges()   {return fNedges;}
   virtual Int_t         GetNsegments() {return fNedges;}     
   Double_t              DistToOutSect(Double_t *point, Double_t *dir, Int_t &iz, Int_t &isect);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   Double_t              DistToInSect(Double_t *point, Double_t *dir, Int_t &iz, Int_t &ipsec,
                                      UChar_t &bits, Double_t *saf); 
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          DefineSection(Int_t snum, Double_t z, Double_t rmin, Double_t rmax);
   virtual void          Draw(Option_t *option);
   virtual void          InspectShape();
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoPgon, 1)         // polygone class 
};

#endif
