// @(#)root/g3d:$Id$
// Author: Nenad Buncic   17/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTRD1
#define ROOT_TTRD1


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTRD1                                                                  //
//                                                                        //
// TRD1 is a trapezoid with only the x length varying with z. It has 4    //
// parameters, the half length in x at the low z surface, that at the     //
// high z surface, the half length in y, and in z.                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TBRIK.h"

class TTRD1 : public TBRIK {
protected:
   Float_t fDx2;        // half length in x at the high z surface

   virtual void    SetPoints(Double_t *points) const;

public:
   TTRD1();
   TTRD1(const char *name, const char *title, const char *material, Float_t dx1, Float_t dx2, Float_t dy, Float_t dz);
   virtual ~TTRD1();

   virtual Float_t GetDx2() const {return fDx2;}

   ClassDef(TTRD1,1)  //TRD1 shape
};

#endif
