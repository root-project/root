// @(#)root/g3d:$Id$
// Author: Nenad Buncic   13/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTRD2
#define ROOT_TTRD2


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTRD2                                                                //
//                                                                      //
// TRD2 is a trapezoid with both x and y lengths varying with z. It     //
// has 5 parameters, half length in x at the low z surface, half length //
// in x at the high z surface, half length in y at the low z surface,   //
// half length in y at the high z surface, and half length in z         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBRIK.h"

class TTRD2 : public TBRIK {
protected:
   Float_t fDx2;        // half length in x at the high z surface
   Float_t fDy2;        // half length in y at the high z surface

   virtual void    SetPoints(Double_t *points) const;

public:
   TTRD2();
   TTRD2(const char *name, const char *title, const char *material, Float_t dx1, Float_t dx2,
         Float_t dy1, Float_t dy2, Float_t dz);
   virtual ~TTRD2();

   Float_t         GetDx2() const {return fDx2;}
   Float_t         GetDy2() const {return fDy2;}

   ClassDef(TTRD2,1)  //TRD2 shape
};

#endif
