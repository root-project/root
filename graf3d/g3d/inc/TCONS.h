// @(#)root/g3d:$Id$
// Author: Nenad Buncic   18/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCONS
#define ROOT_TCONS


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TCONS                                                                  //
//                                                                        //
// CONS is a phi segment of a conical tube. It has 7 parameters, the half //
// the same 5 as a CONE plus the phi limits                               //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTUBS
#include "TTUBS.h"
#endif

class TCONS : public TTUBS {

protected:
   Float_t fRmin2;        // inside radius at the high z limit
   Float_t fRmax2;        // outside radius at the high z limit

   virtual void    SetPoints(Double_t *points) const;
public:
   TCONS();
   TCONS(const char *name, const char *title, const char *material, Float_t dz, Float_t rmin1, Float_t rmax1,
         Float_t rmin2, Float_t rmax2, Float_t phi1, Float_t phi2);
   TCONS(const char *name, const char *title, const char *material, Float_t rmax1, Float_t dz
                          , Float_t phi1, Float_t phi2, Float_t rmax2 = 0);
   virtual ~TCONS();

   virtual Float_t GetRmin2() const {return fRmin2;}
   virtual Float_t GetRmax2() const {return fRmax2;}

   ClassDef(TCONS,1)  //CONS shape
};

#endif
