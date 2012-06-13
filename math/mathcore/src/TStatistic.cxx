// @(#)root/base:$Id$
// Author: G. Ganis 2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStatistic                                                           //
//                                                                      //
// Statistical variable, defined by its mean and RMS.                   //
// Named, streamable, storable and mergeable.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TStatistic.h"


templateClassImp(TStatistic)

//______________________________________________________________________________
TStatistic::TStatistic(const char *name, Int_t n, const Double_t *val, const Double_t *w)
         : fName(name), fN(0), fW(0.), fW2(0.), fMean(0.), fM2(0.)
{
   // Constructor from a vector of values
   
   if (n > 0) {
      for (Int_t i = 0; i < n; i++) {
         if (w) {
            Fill(val[i], w[i]);
         } else {
            Fill(val[i]);
         }
      }
   }   
}
