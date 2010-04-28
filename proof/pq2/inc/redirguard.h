// @(#)root/proof:$Id$
// Author: G. Ganis, Mar 2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef PQ2_redirguard
#define PQ2_redirguard

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// redirguard                                                        //
//                                                                      //
// Auxilliary class used in PQ2 functions to redirect the logs          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSystem.h"

class redirguard {
private:
   RedirectHandle_t fRH;
   Bool_t           fDoIt;
public:
   redirguard(const char *fn, const char *mode = "a", Int_t doit = 0)
       { fDoIt = (doit == 0) ? kTRUE : kFALSE; 
         if (fDoIt) gSystem->RedirectOutput(fn, mode, &fRH); }
   ~redirguard() { if (fDoIt) gSystem->RedirectOutput(0, 0, &fRH); }
};
#endif
