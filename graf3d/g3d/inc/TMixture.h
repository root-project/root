// @(#)root/g3d:$Id$
// Author: Rene Brun   03/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMixture                                                             //
//                                                                      //
// Mixtures used in the Geometry Shapes                                 //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMixture
#define ROOT_TMixture

#ifndef ROOT_TMaterial
#include "TMaterial.h"
#endif

class TMixture  : public TMaterial {
protected:
   Int_t        fNmixt;       //Number of elements in mixture
   Float_t      *fAmixt;      //[fNmixt] Array of A of mixtures
   Float_t      *fZmixt;      //[fNmixt] Array of Z of mixtures
   Float_t      *fWmixt;      //[fNmixt] Array of relative weights

public:
   TMixture();
   TMixture(const char *name, const char *title, Int_t nmixt);
   virtual ~TMixture();

   virtual void  DefineElement(Int_t n, Float_t a, Float_t z, Float_t w);
   Int_t         GetNmixt() const {return fNmixt;}
   Float_t      *GetAmixt() const {return fAmixt;}
   Float_t      *GetZmixt() const {return fZmixt;}
   Float_t      *GetWmixt() const {return fWmixt;}

   ClassDef(TMixture,1)  //Mixtures used in the Geometry Shapes
};

#endif
