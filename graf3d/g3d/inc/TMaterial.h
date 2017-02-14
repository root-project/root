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
// TMaterial                                                            //
//                                                                      //
// Materials used in the Geometry Shapes                                //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMaterial
#define ROOT_TMaterial

#include "TNamed.h"
#include "TAttFill.h"

class TMaterial  : public TNamed, public TAttFill {
protected:
   Int_t        fNumber;      //Material matrix number
   Float_t      fA;           //A of Material
   Float_t      fZ;           //Z of Material
   Float_t      fDensity;     //Material density in gr/cm3
   Float_t      fRadLength;   //Material radiation length
   Float_t      fInterLength; //Material interaction length

public:
   TMaterial();
   TMaterial(const char *name, const char *title, Float_t a, Float_t z, Float_t density);
   TMaterial(const char *name, const char *title, Float_t a, Float_t z, Float_t density, Float_t radl, Float_t inter);
   virtual ~TMaterial();
   virtual Int_t     GetNumber() const      {return fNumber;}
   virtual Float_t   GetA() const           {return fA;}
   virtual Float_t   GetZ() const           {return fZ;}
   virtual Float_t   GetDensity() const     {return fDensity;}
   virtual Float_t   GetRadLength() const   {return fRadLength;}
   virtual Float_t   GetInterLength() const {return fInterLength;}

   ClassDef(TMaterial,3)  //Materials used in the Geometry Shapes
};

#endif
