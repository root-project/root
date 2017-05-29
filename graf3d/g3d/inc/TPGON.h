// @(#)root/g3d:$Id$
// Author: Nenad Buncic   29/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPGON
#define ROOT_TPGON


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TPGON                                                                  //
//                                                                        //
// PGON is a polygone. It has at least 10 parameters, the lower phi limit,//
// the range in phi, the number of straight sides (of equal length)       //
// between those phi limits, the number (at least two) of z planes where  //
// the radius is changing for each z boundary and the z coordinate, the   //
// minimum radius and the maximum radius.                                 //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TPCON.h"

class TPGON : public TPCON {
protected:
   virtual void    FillTableOfCoSin(Double_t phi, Double_t angstep,Int_t n) const; // Fill the table of cosin

public:
   TPGON();
   TPGON(const char *name, const char *title, const char *material, Float_t phi1, Float_t dphi1,
         Int_t npdv, Int_t nz);
   virtual ~TPGON();
   ClassDef(TPGON,1)  //PGON shape
};

#endif
